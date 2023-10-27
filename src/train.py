# pylint: disable=[C0115, C0116, C0301, C0302, W0105]

from functools import partial
import math
from pathlib import Path
import os
import sys
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs

from models.models import (
    HaarModel,
    WordEmbeddings,
)
from systems.recognition_module import get_word_map
from systems.data_module import EpicActionRecognitionDataModule
from utils import get_loggers
from constants import NUM_NOUNS, NUM_VERBS, TRAIN_LOGNAME, DEFAULT_ARGS, DEFAULT_OPT
import datasets as hf_datasets

LOG = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    # TODO: add save_snapshot function
    def __init__(self, name, cfg, device):
        self.cfg = cfg
        self.name = name
        self.device = device
        self.distributed = False

        # initialize systems
        datamodule = EpicActionRecognitionDataModule(cfg)
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        # self.test_loader = datamodule.test_dataloader()
        self.loggers = self._init_loggers()
        self.early_stopping = cfg.trainer.get("early_stopping", None)
        self.use_amp = False if cfg.trainer.get("precision", "full") == "full" else True
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.trainer.get(
                "gradient_accumulation_steps", 1
            ),
            device_placement=True,
            project_dir=self.cfg.model.save_path,
            mixed_precision=self.cfg.trainer.precision,
            log_with="tensorboard",
            kwargs_handlers=[ddp_kwargs],
        )
        # if hf_datasets.logging.is_progress_bar_enabled():
        #     self.accelerator.print("Disabling progress bars for dataset preprocessing")
        #     hf_datasets.logging.disable_progress_bar()

        self.action_embeddings = self.get_embeddings()
        self.model = HaarModel(
            cfg,
            self.action_embeddings,
            cfg.model.transformer.dropout,
            device,
            (
                cfg.model.linear_out.get("verbs", NUM_VERBS),
                cfg.model.linear_out.get("nouns", NUM_NOUNS),
            ),
            embed_size=cfg.model.transformer.d_model,
        )
        self.optimizer = None #self.get_optimizer()
        self.lr_scheduler = None #self.get_lr_scheduler()
        self.loss = self.get_loss_fn()
        self.verb_loss = self.get_loss_fn()
        self.noun_loss = self.get_loss_fn()
        self.cls_head = nn.Softmax(dim=-1)

        LOG = self.loggers["logger"]
        create_snapshot_paths(Path(cfg.model.save_path), logger=LOG)
        LOG.info("Models initialized")
        self.accelerator.print(
            "Trainable parameters = %s",
            {sum([p.numel() for p in self.model.parameters() if p.requires_grad])},
        )

    def _init_loggers(self):
        tb_loc = self.cfg.trainer.get("tb_runs", False)
        tb_writer = None
        if tb_loc:
            os.makedirs(tb_loc, exist_ok=True)
            tb_writer = SummaryWriter(tb_loc)

        logger_name = self.name + ":" + __name__
        logger_filename = self.cfg.trainer.get("logfile", TRAIN_LOGNAME)

        return {
            "logger": get_loggers(logger_name, logger_filename),
            "tb_writer": tb_writer,
        }

    def get_loss_fn(self):
        return nn.CrossEntropyLoss(
            label_smoothing=self.cfg.trainer.loss_fn.args.get("label_smoothing", 0)
        )

    def get_lr_scheduler(self):
        if "lr_scheduler" not in self.cfg.learning:
            return None
        assert self.optimizer is not None, "optimizer not assigned"
        lrs = getattr(
            sys.modules["torch.optim.lr_scheduler"], self.cfg.learning.lr_scheduler.type
        )
        args = self.cfg.learning.lr_scheduler.get("args", {})
        return lrs(self.optimizer, **args)

    def get_optimizer(self, scale_lr=False):
        assert self.model is not None, "model not assigned"
        lr = self.cfg.learning.base_lr
        if scale_lr:
            lr = math.sqrt(self.accelerator.num_processes) * lr
        opt_params = [p for p in self.model.parameters() if p.requires_grad]

        # optimizer_map = {
        #     "Adam": partial(Adam, opt_params, lr=lr),
        #     "AdamW": partial(AdamW, opt_params, lr=lr),
        #     "SGD": partial(SGD, opt_params, lr=lr),
        # }

        if "optimizer" in self.cfg.learning:
            opt_key = self.cfg.learning.optimizer.type
            args = self.cfg.learning.optimizer.get("args", [])
        else:
            opt_key = DEFAULT_OPT
            args = DEFAULT_ARGS
        return getattr(sys.modules["torch.optim"], opt_key)(opt_params, lr=lr, **args) 
        # optimizer_map[opt_key](**args)

    def get_embeddings(self):
        model = WordEmbeddings(device=self.device)
        verb_map = get_word_map(self.cfg.data.verb_loc)
        noun_map = get_word_map(self.cfg.data.noun_loc)

        assert NUM_VERBS == len(
            verb_map
        ), f"NUM_VERBS != len(verb_map) [{NUM_VERBS != len(verb_map)}]"
        assert NUM_NOUNS == len(
            noun_map
        ), f"NUM_NOUNS != len(noun_map) [{NUM_NOUNS != len(noun_map)}]"

        text = verb_map["key"].values.tolist()
        text.extend(noun_map["key"].values.tolist())
        return model(text)

    def _distributed_setup(self):
        if self.lr_scheduler is not None:
            self.train_loader, self.val_loader, self.model, self.optimizer, self.lr_scheduler = \
            self.accelerator.prepare(
                self.train_loader, self.val_loader, self.model, self.optimizer, self.lr_scheduler
            )
        else:
            self.train_loader, self.val_loader, self.model, self.optimizer = \
            self.accelerator.prepare(
                self.train_loader, self.val_loader, self.model, self.optimizer
            )
        
        tb_project_name = self.cfg.trainer.tb_runs.split('/')[-1]
        self.accelerator.init_trackers(project_name=tb_project_name)
        self.distributed = self.accelerator.use_distributed

    def save_snapshot(self, epoch):
        pass

    def _backprop(self, loss):
        assert self.optimizer is not None, "optimizer not initialized"
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def write_to_tensorboard(
        self,
        tb_writer,
        func_name,
        args,
        on_main_process=True,
        gpu_train_type="accelerate",
    ):
        if tb_writer is None:
            return

        tb_function = getattr(tb_writer, func_name)
        if on_main_process:
            if gpu_train_type == "accelerate":
                state = AcceleratorState()
                if state.is_main_process:
                    return tb_function(*args)
            elif gpu_train_type == "ddp":
                if dist.get_rank() == 0:
                    return tb_function(*args)

        return tb_function(*args)

    def num_correct_preds(self, logits, labels, distributed=True):
        if distributed:
            all_preds, all_labels = self.accelerator.gather_for_metrics((logits, labels))
            return (all_preds == all_labels).long().sum().item(), len(all_preds)
        return (logits == labels).long().sum().item(), len(logits)

    #! Will probably deprecate            
    def compute_accuracy(self, logits, labels, distributed=True):
        # y = torch.argmax(self.cls_head(logits), dim=-1)
        if distributed:
            all_preds, all_labels = self.accelerator.gather_for_metrics((logits, labels))
            return (all_preds == all_labels).long().sum().item()
        else:
            batch_labels = labels
        accurate_preds = batch_labels == logits
        # print(accurate_preds)
        return accurate_preds

    def compute_loss(self, verb_logits, verb_labels, noun_logits, noun_labels):
        if self.cfg.trainer.verb and self.cfg.trainer.noun:
            return (
                NUM_VERBS / 2 * self.verb_loss(verb_logits, verb_labels)
                + NUM_NOUNS / 2 * self.noun_loss(noun_logits, noun_labels)
            ) / (NUM_VERBS + NUM_NOUNS)

        if self.cfg.trainer.noun:
            loss = NUM_NOUNS / 2 * self.noun_loss(noun_logits, noun_labels)
        elif self.cfg.trainer.verb:
            loss = NUM_VERBS / 2 * self.verb_loss(verb_logits, verb_labels)
        else:
            raise ValueError("loss could not be computed")

        return loss / (NUM_NOUNS + NUM_VERBS)

    def train(self, num_epochs, validate_strategy="epoch") -> None:
        """Launches the training loop. Based on the config provided,
        it uses either HuggingFace Accelerate or PytTorch DDP to
        perform distributed training and inference.

        __args__
            num_epochs (int): number of epochs to train for
            validate_strategy (str): whether to validate after a
                certain number of steps or after each epoch. Defaults
                to 'epoch'

        """
        # * Determine GPU training strategy
        gpu_strategy = self.cfg.trainer.gpu_training.type
        if gpu_strategy == "single":
            self.model.to(self.device)
        elif gpu_strategy == "accelerate":
            assert self.accelerator is not None, "Accelerator is not initialized"
            self.device = self.accelerator.device
        # TODO elif gpu_strategy == 'ddp':
        #     self._run_ddp(train_ddp, fn_args)

        if validate_strategy == "steps":
            assert (
                "val_every_n_steps" in self.cfg.trainer.validate
            ), "Please specify the validation interval in the config file"

        def accelerate_debug():
            self.accelerator.print("Model named parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.accelerator.print(name, param.shape, param.dtype)

        def check_early_stopping(val_history, early_stop_args):
            assert (
                self.early_stopping is not None
            ), "Early stopping not initialized in config"
            if self.early_stopping["criterion"] == "accuracy":
                verb_history, noun_history = list(zip(*val_history))
                count = 0
                if self.is_early_stopping(verb_history, *early_stop_args):
                    LOG.info("Early stopping threshold reached for verbs.")
                    count += 1
                if self.is_early_stopping(noun_history, *early_stop_args):
                    LOG.info("Early stopping threshold reached for nouns.")
                    count += 1
                if count == 2:
                    LOG.info("Stopping training....")
                    return True
            elif self.early_stopping["criterion"] == "loss" and self.is_early_stopping(
                val_history, *early_stop_args
            ):
                LOG.info("Early stopping threshold reached. Stopping training....")
                return True

            return False
        
        # * Prepare for model training
        tb_writer = self.loggers["tb_writer"]
        LOG = self.loggers["logger"]
        log_n_steps = self.cfg.trainer.log_every_n_steps
        val_n_steps = (
            None
            if validate_strategy == "epoch"
            else self.cfg.trainer.validate.val_every_n_steps
        )
        val_loss_history, val_acc_history = [], []  # list of tuples

        LOG.info("Starting training with config\n %s", OmegaConf.to_yaml(self.cfg))
        self.model.train()
        # if self.accelerator is not None:
        #     self.model.device = self.accelerator.device

        # self.model.to(self.accelerator.device)
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        # print(f'num process = {self.accelerator.num_processes}')
        self._distributed_setup()
        # accelerate_debug()
        # verb_metrics = evaluate.load("accuracy")
        # noun_metrics = evaluate.load("accuracy")

        @torch.no_grad()
        def _validate():
            num_verbs = num_nouns = 0
            correct_preds_verbs = correct_preds_nouns = 0
            val_loss = 0
            # assert self.optimizer is not None, "optimizer not initialized"
            for i, batch in tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                disable=not self.accelerator.is_main_process,
            ):
                # batch.to(self.accelerator.device)
                # self.accelerator.print('val batch on', batch[0][0].device)
                verb_labels = batch[0][1]["verb_class"]
                noun_labels = batch[0][1]["noun_class"]

                verbs_logits, nouns_logits = self.model(
                    batch,
                    # task='inference', # returns class probabilities
                    fp16=self.use_amp,
                    verb=self.cfg.trainer.verb,
                    noun=self.cfg.trainer.noun,
                )
                loss = self.compute_loss(verbs_logits, verb_labels, nouns_logits, noun_labels)
                val_loss += loss.item()
                
                if self.cfg.trainer.verb:
                    preds_verbs = self.cls_head(verbs_logits).argmax(dim=-1)
                    verb_corr, n_v = self.num_correct_preds(preds_verbs, verb_labels, self.distributed)
                    correct_preds_verbs += verb_corr
                    num_verbs += n_v
                if self.cfg.trainer.noun:
                    preds_nouns = self.cls_head(nouns_logits).argmax(dim=-1)
                    noun_corr, n_n = self.num_correct_preds(preds_nouns, noun_labels, self.distributed)
                    correct_preds_nouns += noun_corr
                    num_nouns += n_n
            
            verb_accuracy, noun_accuracy = 0.0, 0.0
            if num_verbs > 0:
                verb_accuracy = correct_preds_verbs / num_verbs
            if num_nouns > 0:
                noun_accuracy = correct_preds_nouns / num_nouns

           
            return (val_loss / len(self.val_loader), verb_accuracy, noun_accuracy)


                # if self.cfg.trainer.verb:
                #     preds_verbs = torch.argmax(preds_verbs, dim=-1)
                #     preds_verbs, refs_verbs = self.accelerator.gather_for_metrics((preds_verbs, verb_labels))
                #     accurate_preds_verb = preds_verbs == refs_verbs
                #     nv += len(accurate_preds_verb)  # type: ignore
                #     correct_preds_verbs += accurate_preds_verb.long().sum().item()
                #     # verb_metrics.add_batch(predictions=preds, references=refs)

                # if self.cfg.trainer.noun:
                #     preds_nouns = torch.argmax(preds_nouns, dim=-1)
                #     preds_nouns, refs_nouns = self.accelerator.gather_for_metrics((preds_nouns, noun_labels))
                #     accurate_preds_noun = preds_nouns == refs_nouns
                #     nn += len(accurate_preds_noun)  # type: ignore
                #     correct_preds_nouns += accurate_preds_noun.long().sum().item()  
                #     # noun_metrics.add_batch(references=refs, predictions=preds)

        # Training loop
        step_counter = 0
        best_val_acc = [0.0, 0.0]
        pbar = tqdm(
            range(len(self.train_loader) * num_epochs),
            disable=not self.accelerator.is_main_process,
        )
        for epoch in range(num_epochs):
            self.accelerator.print(
                f"\n---------------- Epoch: {epoch + 1} ----------------"
            )
            epoch_loss = 0.0
            num_verbs = num_nouns = 0
            # losses = {
            #     'train_loss': torch.zeros(len(self.train_loader)), 
            #     'val_loss': torch.zeros(len(self.val_loader), 1),
            # }
            # losses = self.accelerator.prepare(losses)
            correct_verbs = correct_nouns = 0
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                step_counter += 1
                verb_labels = batch[0][1]["verb_class"]
                noun_labels = batch[0][1]["noun_class"]
               
                with self.accelerator.accumulate(self.model):
                    verb_logits, noun_logits = self.model(
                        batch,
                        fp16=self.use_amp,
                        verb=self.cfg.trainer.verb,
                        noun=self.cfg.trainer.noun,
                    )
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0,
                        )
                    loss = self.compute_loss(
                        verb_logits, verb_labels, noun_logits, noun_labels
                    )
                    epoch_loss += loss.item()
                    # losses['train_loss'][i] = loss.item()
                    self._backprop(loss)

                if self.cfg.trainer.verb:
                    preds_verbs = self.cls_head(verb_logits).argmax(dim=-1)
                    _correct_verbs, _n_verbs = self.num_correct_preds(preds_verbs, verb_labels)
                    correct_verbs += _correct_verbs
                    num_verbs += _n_verbs
                if self.cfg.trainer.noun:
                    preds_nouns = self.cls_head(noun_logits).argmax(dim=-1)
                    _correct_nouns, _n_nouns = self.num_correct_preds(preds_nouns, noun_labels)
                    correct_nouns += _correct_nouns
                    num_nouns += _n_nouns

                if val_n_steps is not None and step_counter % val_n_steps == 0 and self.accelerator.sync_gradients:
                    self.model.eval()
                    (val_loss, val_accuracy_verb, val_accuracy_noun) = _validate() 
                    val_acc_history.append((val_accuracy_verb, val_accuracy_noun))
                    # val_loss = losses['val_loss'].mean()
                    val_loss_history.append(val_loss)
                    if val_accuracy_verb > best_val_acc[0]:
                        self.accelerator.wait_for_everyone()
                        self.accelerator.save_state(
                            os.path.join(self.cfg.model.save_path, "verbs")
                        )
                        best_val_acc[0] = val_accuracy_verb  # type: ignore
                    if val_accuracy_noun > best_val_acc[1]:
                        self.accelerator.wait_for_everyone()
                        self.accelerator.save_state(
                            os.path.join(self.cfg.model.save_path, "nouns")
                        )
                        best_val_acc[1] = val_accuracy_noun  # type: ignore

                    self.accelerator.log(
                        {
                            "val loss": val_loss,
                            "val accuracy [verbs]": val_accuracy_verb,
                            "val accuracy [nouns]": val_accuracy_noun,
                        },
                        step=step_counter,
                    )
                    self.model.train()
                pbar.update(1)

            # Validation loop after each epoch
            if validate_strategy == "epoch":
                self.model.eval()
                val_loss, val_accuracy_verb, val_accuracy_noun = _validate()
                val_acc_history.append((val_accuracy_verb, val_accuracy_noun))
                val_loss_history.append(val_loss)
                if val_accuracy_verb > best_val_acc[0]:
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save_state(
                        os.path.join(self.cfg.model.save_path, "verbs")
                    )
                    best_val_acc[0] = val_accuracy_verb  
                if val_accuracy_noun > best_val_acc[1]:
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save_state(
                        os.path.join(self.cfg.model.save_path, "nouns")
                    )
                    best_val_acc[1] = val_accuracy_noun
                self.accelerator.log(
                    {
                        "val loss": val_loss,
                        "val accuracy [verbs]": val_accuracy_verb,
                        "val accuracy [nouns]": val_accuracy_noun,
                    },
                    step=step_counter,
                )
                self.accelerator.print({
                    "train epoch loss": epoch_loss, 
                    "val accuracy [verbs]": val_accuracy_verb, 
                    "val accuracy [nouns]": val_accuracy_noun,
                })
                self.model.train()

            # Log to tensorboard after each epoch
            avg_loss = epoch_loss / len(self.train_loader)
            # avg_loss = self.accelerator.gather_for_metrics(losses['train_loss']).mean().item()
            avg_verb_acc = correct_verbs / num_verbs if self.cfg.trainer.verb else 0.0
            avg_noun_acc = correct_nouns / num_nouns if self.cfg.trainer.noun else 0.0
            self.accelerator.log(
                {
                    "train loss": avg_loss,
                    "train accuracy [verbs]": avg_verb_acc, 
                    "train accuracy [nouns]": avg_noun_acc,
                },
                step = step_counter
            )

            # Check early stopping conditions after each epoch
            if self.early_stopping is not None:
                patience = self.early_stopping["epochs"]
                threshold_epoch = self.early_stopping["threshold_epochs"]
                threshold_value = self.early_stopping["threshold"]

                if self.early_stopping[
                    "criterion"
                ] == "accuracy" and check_early_stopping(
                    val_acc_history,
                    early_stop_args=(patience, threshold_epoch, threshold_value),
                ):
                    break

                if self.early_stopping[
                    "criterion"
                ] == "loss" and check_early_stopping(
                    val_loss_history,
                    early_stop_args=(patience, threshold_epoch, threshold_value),
                ):
                    break

            # * Step LR scheduler after each epoch
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


        self.accelerator.wait_for_everyone()
        LOG.info("Training completed!")
        self.accelerator.end_training()

    def is_early_stopping(
        self, metrics, early_stopping_epochs, threshold_epochs=0, threshold_val=0.0
    ):
        """metrics should be in order"""
        # Too few observations
        if len(metrics) <= early_stopping_epochs or len(metrics) < threshold_epochs:
            return False

        last_n_metrics = metrics[-early_stopping_epochs:]
        prev_n_metrics = metrics[-early_stopping_epochs - 1 : -1]
        if np.allclose(prev_n_metrics, last_n_metrics) and all(
            met for met in last_n_metrics > threshold_val
        ):
            return True

        return False


def debug(loader):
    for item in loader:
        rgb, flow = item
        rgb_tensors, rgb_metadata = rgb
        print(len(item))
        print(rgb_tensors.size())
        print(rgb_metadata.keys())
        break


def set_env_ddp(nccl_p2p_disable):
    # required unless ACS is disabled on host. Ref: https://github.com/NVIDIA/nccl/issues/199
    if nccl_p2p_disable == "1":
        os.environ["NCCL_P2P_DISABLE"] = "1"
        print("Disabled P2P transfer for NCCL")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def create_snapshot_paths(model_save_path, logger=None):
    verb_save_path = model_save_path / "verbs"
    noun_save_path = model_save_path / "nouns"
    verb_save_path.mkdir(parents=True, exist_ok=True)
    noun_save_path.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.info("Created snapshot paths for verbs and nouns")
    return verb_save_path, noun_save_path


def run_ddp(system, datamodule, port, verb_save_path, noun_save_path, ddp_fn):
    world_size = system.cfg.trainer.gpus
    batch_size = int(system.cfg.learning.batch_size)
    num_epochs = system.cfg.trainer.max_epochs
    nprocs = system.cfg.learning.ddp.nprocs
    if nprocs == "gpu":
        nprocs = world_size
    # print(f'mp.spawn args - world_size={world_size}, nprocs={nprocs}')
    mp.spawn(
        ddp_fn,
        args=(
            world_size,
            port,
            datamodule,
            system,
            verb_save_path,
            noun_save_path,
            batch_size,
            num_epochs,
        ),
        nprocs=nprocs,
        join=True,
    )


def ddp_setup(rank, world_size, master_port, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    init_process_group(backend, rank=rank, world_size=world_size)

def get_device_from_config(config):
    if "gpu_training" not in config.trainer:
        return torch.device("cpu")
    training_type = config.trainer.gpu_training.type
    if training_type == "None":
        return torch.device("cpu")
    elif training_type == "single":
        return torch.device(config.trainer.gpu_training.args.device_id)
    elif training_type == "accelerate":
        return None
    else:
        raise Exception("Incorrect config mapping")


@hydra.main(config_path="../configs", config_name="pilot_config", version_base=None)
def main(cfg: DictConfig):
    # LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device_from_config(cfg)
    trainer = Trainer(name="transformers_testing", cfg=cfg, device=device)  # type: ignore
    validate_strategy = "epoch" if cfg.trainer.validate == "epoch" else "steps"
    trainer.train(cfg.trainer.max_epochs, validate_strategy=validate_strategy)
    return 0

if __name__ == "__main__":
    sys.exit(main())

''' INFO: args to run on usha:

trainer.gpus=6 \
learning.ddp.nprocs=6 \
learning.lr=0.1 \
learning.batch_size=2 \
learning.val_batch_size=8
'''