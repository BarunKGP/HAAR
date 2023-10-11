from functools import partial
from pathlib import Path
import sys
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import numpy as np


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD

from accelerate import Accelerator
from accelerate.state import AcceleratorState

from models.models import (
    AttentionModel,
    EmbeddingModel,
    HaarModel,
    WordEmbeddings,
)  # pylint: disable=unused-import
from systems.recognition_module import get_word_map
from systems.data_module import EpicActionRecognitionDataModule
from utils import (
    ActionMeter,
    get_loggers,
    log_print,
    write_pickle,
)
from constants import NUM_NOUNS, NUM_VERBS, TRAIN_LOGNAME, DEFAULT_ARGS, DEFAULT_OPT
import datasets as hf_datasets

LOG = get_loggers(__name__, filename=TRAIN_LOGNAME)


class Trainer:
    # pylint: disable=C0116
    # TODO: add save_snapshot function
    def __init__(self, name, cfg, device):
        self.cfg = cfg
        self.name = name
        self.device = device

        # initialize systems
        datamodule = EpicActionRecognitionDataModule(cfg)
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        # self.test_loader = datamodule.test_dataloader()
        self.loggers = self._init_loggers()
        self.early_stopping = cfg.trainer.get("early_stopping", None)
        self.use_amp = False if cfg.trainer.get("precision", "full") == "full" else True
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.trainer.get(
                "gradient_accumulation_steps", 1
            ),
            project_dir=self.cfg.model.save_path,
            log_with="tensorboard",
        )

        self.action_embeddings = self.get_embeddings()
        self.accelerator.print(f"action embeddings -> {self.action_embeddings.shape}")
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
        self.optimizer = self.get_optimizer()
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

    def get_optimizer(self):
        assert self.model is not None, "model not assigned"
        lr = self.cfg.learning.base_lr
        opt_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_map = {
            "Adam": partial(Adam, opt_params, lr=lr),
            "AdamW": partial(AdamW, opt_params, lr=lr),
            "SGD": partial(SGD, opt_params, lr=lr),
        }

        if "optimizer" in self.cfg.learning:
            opt_key = self.cfg.learning.optimizer.type
            args = self.cfg.learning.optimizer.get("args", [])
        else:
            opt_key = DEFAULT_OPT
            args = DEFAULT_ARGS
        return optimizer_map[opt_key](foreach=True, **args)

    def get_embeddings(self):
        model = WordEmbeddings(device=self.device)
        verb_map = get_word_map(self.cfg.data.verb_loc)
        noun_map = get_word_map(self.cfg.data.noun_loc)
        # self.accelerator.print(f'\n len(noun_map) = {len(noun_map)}')
        # self.accelerator.print(noun_map, '\n')

        assert NUM_VERBS == len(
            verb_map
        ), f"NUM_VERBS != len(verb_map) [{NUM_VERBS != len(verb_map)}]"
        assert NUM_NOUNS == len(
            noun_map
        ), f"NUM_NOUNS != len(noun_map) [{NUM_NOUNS != len(noun_map)}]"

        text = verb_map["key"].values.tolist()
        text.extend(noun_map["key"].values.tolist())
        return model(text)

    def _distributed_setup(self) -> None:
        if self.cfg.trainer.gpu_training.type.lower() in ["single", "none"]:
            return None

        elif self.cfg.trainer.gpu_training.type.lower() == "accelerate":
            # accelerator = Accelerator(gradient_accumulation_steps=self.cfg.trainer.get('gradient_accumulation_steps', 1))
            (
                self.train_loader,
                self.val_loader,
                self.model,
                self.optimizer,
            ) = self.accelerator.prepare(
                self.train_loader, self.val_loader, self.model, self.optimizer
            )
            self.accelerator.init_trackers(
                project_name="tb",
                # init_kwargs={'tensorboard': {'logging_dir': self.cfg.trainer.tb_runs}}
            )

        elif self.cfg.trainer.gpu_training.type.lower() == "ddp":
            set_env_ddp(
                nccl_p2p_disable=self.cfg.trainer.gpu_training.args.get(
                    "nccl_p2p_disable", "0"
                )
            )

        else:
            assert (
                False
            ), "Invalid GPU training strategy. Please choose among [single, accelerate, ddp, None]"

    def save_snapshot(self, epoch):
        pass

    def _run_ddp(self, ddp_fn, fn_args):
        ddp_args = self.cfg.trainer.gpu_training.args
        init_process_group(ddp_args.backend)
        mp.spawn(
            ddp_fn,
            args=(fn_args),
            nprocs=ddp_args.nprocs,
            join=True,
        )

    def _backprop(self, loss):
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # # self.scaler.scale(loss)
        # if self.use_amp:
        # # if self.accelerator is not None:
        #     self.accelerator.backward(self.scaler.scale(loss))
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        #     self.accelerator.backward(loss)
        #     self.optimizer.step()
        # self.optimizer.zero_grad(set_to_none=True)

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

    def compute_accuracy(self, logits, labels, distributed=True):
        y = torch.argmax(self.cls_head(logits), dim=-1)
        if distributed:
            accurate_preds = self.accelerator.gather(y) == self.accelerator.gather(
                labels
            )
        else:
            accurate_preds = y == labels
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

        # Prepare for model training
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
        if self.accelerator is not None:
            self.model.device = self.accelerator.device

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
                    LOG.info(
                        "Early stopping threshold reached for both verbs and nouns. Stopping training...."
                    )
                    return True
            elif self.early_stopping["criterion"] == "loss":
                if self.is_early_stopping(val_history, *early_stop_args):
                    LOG.info("Early stopping threshold reached. Stopping training...")
                    return True

            return False

        self._distributed_setup()
        if hf_datasets.logging.is_progress_bar_enabled():
            self.accelerator.print("Disabling progress bars for dataset preprocessing")
            hf_datasets.logging.disable_progress_bar()  # disable progress bars during dataset loading and processing
        # accelerate_debug()

        # Training loop
        step_counter = 0
        best_val_loss = 10000
        best_val_acc = [0, 0]
        for epoch in range(num_epochs):
            self.accelerator.print(
                f"\n---------------- Epoch: {epoch + 1} ----------------"
            )
            epoch_loss = 0.0
            step_loss = 0.0
            num_elems = 0
            correct_preds_verb = correct_preds_noun = 0
            for i, batch in tqdm(enumerate(self.train_loader)):
                step_counter += 1
                verb_labels = batch[0][1]["verb_class"]
                noun_labels = batch[0][1]["noun_class"]
                with self.accelerator.accumulate(self.model):
                    verb_logits, noun_logits = self.model(
                        batch, fp16=self.use_amp
                    )  # self._train_helper(batch)
                    loss = self.compute_loss(
                        verb_logits, verb_labels, noun_logits, noun_labels
                    )
                    self._backprop(loss)
                    step_loss += loss.item()

                # if self.cfg.trainer.verb:
                #     accurate_preds_verbs = self.compute_accuracy(verb_logits, verb_labels)
                #     accurate_preds_verbs = accurate_preds_verbs.long().sum().item() # type: ignore
                # else:
                #     accurate_preds_verbs = 0

                # if self.cfg.trainer.noun:
                #     accurate_preds_nouns = self.compute_accuracy(noun_logits, noun_labels)
                #     accurate_preds_nouns = accurate_preds_nouns.long().sum().item() # type: ignore
                # else:
                #     accurate_preds_nouns = 0

                # correct_preds_verb += accurate_preds_verbs
                # correct_preds_noun += accurate_preds_nouns
                # num_elems += max(verb_logits.shape[0], noun_logits.shape[0])

                if val_n_steps is not None and step_counter % val_n_steps == 0:
                    val_accuracy_verb, val_accuracy_noun = self.validate()
                    val_acc_history.append((val_accuracy_verb, val_accuracy_noun))
                    if val_accuracy_verb > best_val_acc[0]:
                        self.accelerator.save_state(
                            os.path.join(self.cfg.model.save_path, "verbs")
                        )
                        best_val_acc[0] = val_accuracy_verb  # type: ignore
                    if val_accuracy_noun > best_val_acc[1]:
                        self.accelerator.save_state(
                            os.path.join(self.cfg.model.save_path, "nouns")
                        )
                        best_val_acc[1] = val_accuracy_noun  # type: ignore

                    # tb_steps = epoch * len(self.train_loader) + i
                    self.accelerator.log(
                        {
                            "val accuracy [verbs]": val_accuracy_verb,
                            "val accuracy [nouns]": val_accuracy_noun,
                        },
                        step=step_counter,
                    )

                # Log training progress to tensorboard
                if step_counter % log_n_steps == 0:
                    avg_loss = step_loss / log_n_steps

                    nv = nn = 0
                    if self.cfg.trainer.verb:
                        accurate_preds_verbs = self.compute_accuracy(
                            verb_logits, verb_labels
                        )
                        nv = len(accurate_preds_verbs)  # type: ignore
                        accurate_preds_verbs = accurate_preds_verbs.long().sum().item()  # type: ignore
                    else:
                        accurate_preds_verbs = 0

                    if self.cfg.trainer.noun:
                        accurate_preds_nouns = self.compute_accuracy(
                            noun_logits, noun_labels
                        )
                        nn = len(accurate_preds_nouns)  # type: ignore
                        accurate_preds_nouns = accurate_preds_nouns.long().sum().item()  # type: ignore
                    else:
                        accurate_preds_nouns = 0

                    num_elems = max(nv, nn)
                    avg_acc_verb = accurate_preds_verbs / num_elems
                    avg_acc_noun = accurate_preds_nouns / num_elems
                    self.accelerator.log(
                        {
                            "train loss": avg_loss,
                            "train accuracy [verbs]": avg_acc_verb,
                            "train accuracy [nouns]": avg_acc_noun,
                        },
                        step=step_counter,
                    )

                    epoch_loss += step_loss
                    step_loss = 0.0
                    correct_preds_verb = correct_preds_noun = 0
                    # step_counter = 0
                    # tb_steps = epoch * len(self.train_loader) + step_counter

                    # self.write_to_tensorboard(
                    #     tb_writer,
                    #     func_name="add_scalars",
                    #     args=(
                    #         "Training progress",
                    #         {
                    #             "train loss": avg_loss,
                    #             "train accuracy [verbs]": avg_acc_verb,
                    #             "train accuracy [nouns]": avg_acc_noun,
                    #         },
                    #         tb_steps,
                    #     ),
                    #     gpu_train_type=gpu_strategy,
                    # )

            # Validation loop after each epoch
            if validate_strategy == "epoch":
                val_accuracy_verb, val_accuracy_noun = self.validate(best_val_loss)
                tb_steps = (epoch + 1) * len(self.train_loader)
                self.write_to_tensorboard(
                    tb_writer,
                    func_name="add_scalars",
                    args=(
                        "Validation Accuracy",
                        {
                            "val accuracy [verbs]": val_accuracy_verb,
                            "val accuracy [nouns]": val_accuracy_noun,
                        },
                        tb_steps,
                    ),
                    gpu_train_type=gpu_strategy,
                )
                # val_loss_history.append(val_loss)
                val_acc_history.append((val_accuracy_verb, val_accuracy_noun))
                if val_accuracy_verb > best_val_acc[0]:
                    self.accelerator.save_state(
                        os.path.join(self.cfg.model.save_path, "verbs")
                    )
                    best_val_acc[0] = val_accuracy_verb  # type: ignore
                if val_accuracy_noun > best_val_acc[1]:
                    self.accelerator.save_state(
                        os.path.join(self.cfg.model.save_path, "nouns")
                    )
                    best_val_acc[1] = val_accuracy_noun  # type: ignore

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

                elif self.early_stopping[
                    "criterion"
                ] == "loss" and check_early_stopping(
                    val_loss_history,
                    early_stop_args=(patience, threshold_epoch, threshold_value),
                ):
                    break

        LOG.info("Training completed!")
        self.accelerator.end_training()

    def validate(self, distributed=True):
        """
        Validation run for a batch. Each example within a batch contains
        two tuples: ((rgb, metadata), (flow, metadata))

        __args__
            distributed: whether to use distributed setup for validation

        __returns__
            val_acc_verb (float): validation accuracy (verbs)
            val_acc_noun (float): validation accuracy (nouns)
        """
        # TODO: configure distributed validation on other training modes (DDP)
        # TODO: figure out non-distributed validation
        self.model.eval()
        correct_preds_verbs = correct_preds_nouns = 0.0
        num_elems = 0
        for batch in tqdm(self.val_loader):
            verb_labels = batch[0][1]["verb_class"]
            noun_labels = batch[0][1]["noun_class"]
            nv = nn = 0
            # with torch.no_grad():
            verb_logits, noun_logits = self.model(batch, fp16=self.use_amp)
            if self.cfg.trainer.verb:
                accurate_preds_verbs = self.compute_accuracy(verb_logits, verb_labels, distributed=distributed)
                nv = len(accurate_preds_verbs)  # type: ignore
                accurate_preds_verbs = accurate_preds_verbs.long().sum().item()  # type: ignore

            else:
                accurate_preds_verbs = 0

            if self.cfg.trainer.noun:
                accurate_preds_nouns = self.compute_accuracy(noun_logits, noun_labels, distributed=distributed)
                nn = len(accurate_preds_nouns)  # type: ignore
                accurate_preds_nouns = accurate_preds_nouns.long().sum().item()  # type: ignore
            else:
                accurate_preds_nouns = 0

            correct_preds_verbs += accurate_preds_verbs
            correct_preds_nouns += accurate_preds_nouns
            num_elems += max(nn, nv)

        #! USE num_elems instead?
        # self.accelerator.print(f'num_elems = {num_elems}, val loader length = {len(self.val_loader)}')
        val_accuracy_verbs = correct_preds_verbs / num_elems
        val_accuracy_nouns = correct_preds_nouns / num_elems
        # val_loss = val_loss / len(self.val_loader)

        return val_accuracy_verbs, val_accuracy_nouns

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


#! should de deprecated but can be used for DDP training in Trainer
def train(
    device,
    datamodule,
    system,
    verb_save_path,
    noun_save_path,
    writer,
):
    system.device = device
    batch_size = system.cfg.learning.batch_size
    val_batch_size = system.cfg.learning.val_batch_size
    num_epochs = system.cfg.trainer.max_epochs
    train_loader = datamodule.train_dataloader(device)
    val_loader = datamodule.val_dataloader(device)
    log_every_n_steps = system.cfg.trainer.get("log_every_n_steps", 1)
    steps_per_run = len(train_loader)
    (
        train_loss_history,
        validation_loss_history,
        train_accuracy_history,
        validation_accuracy_history,
    ) = ([], [], [], [])

    verb_snapshot_path = noun_snapshot_path = None
    if "load_snapshot" in system.cfg.trainer:
        verb_snapshot_path = system.cfg.trainer.load_snapshot.get(
            "verb_snapshot_path", None
        )
        if verb_snapshot_path is not None:
            verb_snapshot_path = Path(verb_snapshot_path)
        noun_snapshot_path = system.cfg.trainer.load_snapshot.get(
            "noun_snapshot_path", None
        )
        if noun_snapshot_path is not None:
            noun_snapshot_path = Path(noun_snapshot_path)

    def train_one_epoch(model, epoch_index):
        running_loss = avg_loss = avg_acc = 0.0
        train_loss_meter = ActionMeter("train loss")
        train_acc_meter = ActionMeter("train accuracy")
        for i, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"train_loader epoch {epoch_index + 1}/{num_epochs}",
                total=steps_per_run,
                leave=False,
            )
        ):
            batch_acc, batch_loss = system._step(batch, model, key)
            avg_loss += batch_loss.item()
            avg_acc += batch_acc
            train_acc_meter.update(batch_acc, batch_size)
            train_loss_meter.update(batch_loss.item(), batch_size)
            system.backprop(model, batch_loss)

            if (i + 1) % log_every_n_steps == 0:
                running_loss += avg_loss
                avg_loss = avg_loss / log_every_n_steps
                avg_acc = (
                    avg_acc / log_every_n_steps
                )  # train_acc_meter.get_average_values()
                # log_print(
                #     LOG,
                #     f'batch: {i + 1} \t loss: {avg_loss} \t accuracy: {avg_acc}'
                # )
                tb_steps = epoch_index * steps_per_run + i + 1
                writer.add_scalars(
                    "Train metrics",
                    {"train loss": avg_loss, "train accuracy": avg_acc},
                    tb_steps,
                )
                # train_loss_meter.reset()
                # train_acc_meter.reset()
                avg_loss = avg_acc = 0.0
                break
        LOG.info(f"train loss meter: {train_loss_meter.get_average_values()}")
        return running_loss / steps_per_run, train_acc_meter.get_average_values()

    def train_loop(
        model, snapshot_save_path, key, tqdm_desc="training loop", epoch_start=0
    ):
        best_val_acc = 0.0
        # log_print(LOG, f"len(train_loader) = {len(train_loader)} = {steps_per_run}")
        # for epoch in tqdm(range(epoch_start, num_epochs), desc=tqdm_desc, position=0):
        for epoch in range(epoch_start, num_epochs):
            model.train(True)
            system.rgb_model.train(True)
            system.flow_model.train(True)
            train_loss, train_acc = train_one_epoch(model, epoch)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_acc)

            # Validation
            model.eval()
            system.rgb_model.eval()
            system.flow_model.eval()
            # val_loss_meter = ActionMeter("val loss")
            val_acc_meter = ActionMeter("val accuracy")
            running_vloss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"val_loader epoch {epoch+1}/{num_epochs}",
                    total=len(val_loader),
                    leave=False,
                ):
                    batch_acc, batch_loss = system._step(batch, model, key)
                    val_acc_meter.update(batch_acc, val_batch_size)
                    running_vloss += batch_loss.item()
                    # val_loss_meter.update(batch_loss.item(), val_batch_size)

            # val_loss = val_loss_meter.get_average_values()
            val_loss = running_vloss / len(val_loader)
            val_acc = val_acc_meter.get_average_values()
            validation_loss_history.append(val_loss)
            validation_accuracy_history.append(val_acc)
            if system.lr_scheduler is not None:
                system.lr_scheduler.step(val_loss)

            log_print(
                LOG,
                f"[Epoch {epoch + 1}/{num_epochs}]"
                + f" Train Loss/Val Loss: {train_loss} / {val_loss}"
                + f"\tTrain Accuracy/Val Accuracy: {train_acc:4f} / {val_acc:.4f}",
                "info",
            )
            log_print(LOG, f"Best val_acc = {best_val_acc}")
            writer.add_scalars(
                tqdm_desc + ": Loss",
                {
                    "train loss": train_loss,
                    "val loss": val_loss,
                },
                steps_per_run * (epoch + 1),
            )
            writer.add_scalars(
                tqdm_desc + ": Accuracy",
                {"train accuracy": train_acc, "val accuracy": val_acc},
                steps_per_run * (epoch + 1),
            )
            saved = system.save_model(
                model, val_acc, best_val_acc, epoch + 1, snapshot_save_path
            )
            if saved:
                LOG.info(
                    f"Saved model state for epoch {epoch + 1} at {snapshot_save_path}/checkpoint_{epoch + 1}.pt"
                )

            model.train()
            system.rgb_model.train()
            system.flow_model.train()
            best_val_acc = max(best_val_acc, val_acc)
            if system.early_stopping(val_loss, val_acc):
                break

    train_verb = system.cfg.trainer.get("verb", True)
    train_noun = system.cfg.trainer.get("noun", True)
    # VERB TRAINING
    if train_verb:
        system.load_models_to_device(device, verb=True)

        (
            train_loss_history,
            validation_loss_history,
            train_accuracy_history,
            validation_accuracy_history,
        ) = ([], [], [], [])

        assert system.verb_embeddings is not None, "verb embeddings not initialized"
        assert system.verb_map is not None, "verb map not initialized"

        verb_model = AttentionModel(system.verb_map).to(device)
        opt_sd = lr_scheduler_sd = None
        epoch_start = 0
        if verb_snapshot_path is not None:
            epoch_start, opt_sd, lr_scheduler_sd = system.load_snapshot(
                verb_snapshot_path, device, verb_model, model_key="attention_model"
            )
            LOG.info(
                f"Loaded state dict for models from {verb_snapshot_path}, starting at epoch {epoch_start}"
            )
        key = "verb_class"
        system.opt = system.get_optimizer(verb_model)
        if opt_sd is not None:
            system.opt.load_state_dict(opt_sd)
            LOG.info(f"Loaded state dict for optimizer from {verb_snapshot_path}")
        system.lr_scheduler = system.get_lr_scheduler()
        if lr_scheduler_sd is not None:
            system.lr_scheduler.load_state_dict(lr_scheduler_sd)
            LOG.info(f"Loaded state dict for LR scheduler from {verb_snapshot_path}")
        log_print(
            LOG,
            "---------------- ### PHASE 1: TRAINING VERBS ### ----------------",
            "info",
        )
        train_loop(
            verb_model, verb_save_path, key, "training loop (verbs)", epoch_start
        )
        train_stats = {
            "train_accuracy": train_accuracy_history,
            "train_loss": train_loss_history,
            "val_accuracy": validation_accuracy_history,
            "val_loss": validation_loss_history,
        }

        # Write training stats for analysis
        fname = os.path.join(system.cfg.model.save_path, "train_stats_verbs.pkl")
        write_pickle(train_stats, fname)
        LOG.info("Finished verb training")

    # NOUN TRAINING
    if train_noun:
        system.load_models_to_device(device, verb=True)
        (
            train_loss_history,
            validation_loss_history,
            train_accuracy_history,
            validation_accuracy_history,
        ) = ([], [], [], [])

        if train_verb:
            system.freeze_feature_extractors()
        key = "noun_class"
        torch.cuda.empty_cache()
        epoch_start = 0
        system.load_models_to_device(device, verb=False)
        noun_model = AttentionModel(system.noun_map).to(device)
        opt_sd = lr_scheduler_sd = None
        if noun_snapshot_path is not None:
            epoch_start, opt_sd, lr_scheduler_sd = system.load_snapshot(
                noun_snapshot_path, device, noun_model, "attention_model"
            )
            LOG.info(
                f"Loaded state dict for models from {noun_snapshot_path}, starting at epoch {epoch_start}"
            )
        system.opt = system.get_optimizer(noun_model)
        if opt_sd is not None:
            system.opt.load_state_dict(opt_sd)
            LOG.info(f"Loaded state dict for optimizer from {noun_snapshot_path}")
        if lr_scheduler_sd is not None:
            system.lr_scheduler.load_state_dict(lr_scheduler_sd)
            LOG.info(f"Loaded state dict for LR scheduler from {noun_snapshot_path}")
        log_print(
            LOG,
            "---------------- ### PHASE 2: TRAINING NOUNS ### ----------------",
            "info",
        )
        train_loop(
            noun_model, noun_save_path, key, "training loop (nouns)", epoch_start
        )
        train_stats = {
            "train_accuracy": train_accuracy_history,
            "train_loss": train_loss_history,
            "val_accuracy": validation_accuracy_history,
            "val_loss": validation_loss_history,
        }
        # Write training stats for analysis
        fname = os.path.join(system.cfg.model.save_path, "train_stats_nouns.pkl")
        write_pickle(train_stats, fname)
        LOG.info("Finished noun training")


def ddp_train(
    rank,
    world_size,
    free_port,
    datamodule,
    system,
    verb_save_path,
    noun_save_path,
    batch_size,
    num_epochs,
):
    ddp_setup(
        rank, world_size, free_port, system.cfg.learning.ddp.backend
    )  # should be taken from cfg
    global_rank = dist.get_rank()
    system.device = torch.device(rank)
    train_loader = datamodule.train_dataloader(rank)
    val_loader = datamodule.val_dataloader(rank)
    log_every_n_steps = system.cfg.trainer.get("log_every_n_steps", 1)
    val_batch_size = system.cfg.learning.val_batch_size
    steps_per_run = len(train_loader)
    writer = SummaryWriter(system.cfg.trainer.tb_runs)
    (
        train_loss_history,
        validation_loss_history,
        train_accuracy_history,
        validation_accuracy_history,
    ) = ([], [], [], [])

    def ddp_one_epoch(model, epoch_index):
        """Trains DDP model for one epoch and logs loss and accuracy
        to Tensorboard.

        Args:
            model: DDP model to train
            epoch_index: current epoch index being trained

        Returns:
            (avg_train_loss, avg_train_acc): Tuple containing average
            training loss and accuracy for that epoch
        """
        running_loss = avg_loss = avg_acc = 0.0
        acc_meter = ActionMeter("accuracy")
        for i, batch in enumerate(train_loader):
            batch_acc, batch_loss = system._step(batch, ddp_model, key)
            avg_loss += batch_loss.item()
            avg_acc += batch_acc
            acc_meter.update(batch_acc, system.cfg.learning.batch_size)
            system.backprop(ddp_model, batch_loss)

            if (i + 1) % log_every_n_steps == 0:
                running_loss += avg_loss
                avg_loss = avg_loss / log_every_n_steps
                avg_acc = (
                    avg_acc / log_every_n_steps
                )  # train_acc_meter.get_average_values()
                tb_steps = epoch_index * steps_per_run + i + 1
                writer.add_scalars(
                    "Real-time training metrics",
                    {"train loss": avg_loss, "train accuracy": avg_acc},
                    tb_steps,
                )
                avg_loss = avg_acc = 0.0

        return running_loss / steps_per_run, acc_meter.get_average_values()

    def ddp_loop(
        ddp_model, snapshot_save_path, key, tqdm_desc="training loop", epoch_start=0
    ):
        best_val_acc = 0.0
        for epoch in tqdm(
            range(epoch_start, num_epochs),
            desc=tqdm_desc,
            position=0,
        ):
            train_loader.sampler.set_epoch(epoch)
            train_loss, train_acc = ddp_one_epoch(ddp_model, epoch)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_acc)

            # Validation
            val_loader.sampler.set_epoch(epoch)
            ddp_model.eval()
            system.rgb_model.eval()
            system.flow_model.eval()
            val_loss_meter = ActionMeter("val loss")
            val_acc_meter = ActionMeter("val accuracy")
            with torch.no_grad():
                running_vloss = 0.0
                num_items = 0
                for batch in val_loader:
                    # for batch in tqdm(
                    #     val_loader,
                    #     desc=f"val_loader epoch: {epoch+1}",
                    #     total=len(val_loader),
                    #     leave=False
                    # ):
                    batch_acc, batch_loss = system._step(batch, ddp_model, key)
                    log_print(LOG, f"num of items = {len(batch[0][0])} on rank {rank}")
                    num_items += len(batch[0][0])
                    val_acc_meter.update(batch_acc, len(batch[0][0]))
                    running_vloss += batch_loss.item()
                    val_loss_meter.update(batch_loss.item(), len(batch[0][0]))
                # val_loss = val_loss_meter.get_average_values()
                print(
                    f"val_loss on rank {rank} = {val_loss_meter.get_average_values()}"
                )
                val_loss = running_vloss / len(val_loader)
                val_acc = val_acc_meter.get_average_values()
                validation_loss_history.append(val_loss)
                validation_accuracy_history.append(val_acc)
                if system.lr_scheduler is not None:
                    system.lr_scheduler.step(val_loss)

            if global_rank == 0:
                log_print(
                    LOG,
                    f"Epoch:{epoch + 1}"
                    + f" Train Loss/Val Loss: {train_loss:4f}/{val_loss:4f}"
                    + f" Train Accuracy/Val Accuracy: {train_acc:4f}/{val_acc:.4f}",
                    "info",
                )
                saved = system.save_model(
                    ddp_model, val_acc, best_val_acc, epoch + 1, snapshot_save_path
                )
                if saved:
                    LOG.info(
                        f"Saved model state for epoch {epoch + 1} at {snapshot_save_path}/checkpoint_{epoch + 1}.pt"
                    )

            writer.add_scalars(
                tqdm_desc + ": Loss",
                {
                    "train loss": train_loss,
                    "val loss": val_loss,
                },
                steps_per_run * (epoch + 1),
            )
            writer.add_scalars(
                tqdm_desc + ": Accuracy",
                {"train accuracy": train_acc, "val accuracy": val_acc},
                steps_per_run * (epoch + 1),
            )
            if system.early_stopping(val_loss, val_acc):
                break
            ddp_model.train()
            system.rgb_model.train()
            system.flow_model.train()
            best_val_acc = max(best_val_acc, val_acc)

    verb_snapshot_path = noun_snapshot_path = None
    if "load_snapshot" in system.cfg.trainer:
        verb_snapshot_path = system.cfg.trainer.load_snapshot.get(
            "verb_snapshot_path", None
        )
        if verb_snapshot_path is not None:
            verb_snapshot_path = Path(verb_snapshot_path)
        noun_snapshot_path = system.cfg.trainer.load_snapshot.get(
            "noun_snapshot_path", None
        )
        if noun_snapshot_path is not None:
            noun_snapshot_path = Path(noun_snapshot_path)
    train_verb = system.cfg.trainer.get("verb", True)
    train_noun = system.cfg.trainer.get("noun", True)

    # VERB TRAINING
    if train_verb:
        system.load_models_to_device(device=rank, verb=True)

        (
            train_loss_history,
            validation_loss_history,
            train_accuracy_history,
            validation_accuracy_history,
        ) = ([], [], [], [])

        assert system.verb_embeddings is not None, "verb embeddings not initialized"
        assert system.verb_map is not None, "verb map not initialized"

        verb_model = AttentionModel(system.verb_map).to(rank)
        opt_sd = lr_scheduler_sd = None
        epoch_start = 0
        if verb_snapshot_path is not None:
            epoch_start, opt_sd, lr_scheduler_sd = system.load_snapshot(
                verb_snapshot_path,
                torch.device(rank),
                verb_model,
                model_key="ddp_model",
            )
            LOG.info(
                f"Loaded state dict for models from {verb_snapshot_path}, starting at epoch {epoch_start}"
            )
        ddp_model = DDP(verb_model, device_ids=[rank])
        key = "verb_class"
        system.opt = system.get_optimizer(ddp_model)
        if opt_sd is not None:
            system.opt.load_state_dict(opt_sd)
            LOG.info(
                f"Loaded state dict for optimizer from {verb_snapshot_path}, starting at epoch {epoch_start}"
            )
        if global_rank == 0:
            log_print(
                LOG,
                "---------------- ### PHASE 1: TRAINING VERBS ### ----------------",
                "info",
            )
        ddp_loop(ddp_model, verb_save_path, key, "training loop (verbs)", epoch_start)
        # dist.barrier()
        train_stats = {
            "train_accuracy": train_accuracy_history,
            "train_loss": train_loss_history,
            "val_accuracy": validation_accuracy_history,
            "val_loss": validation_loss_history,
        }

        # Write training stats for analysis
        fname = os.path.join(system.cfg.model.save_path, "train_stats_verbs.pkl")
        if rank == 0:
            write_pickle(train_stats, fname)
            LOG.info("Finished verb training")

    # NOUN TRAINING
    if train_noun:
        system.load_models_to_device(rank, verb=True)
        (
            train_loss_history,
            validation_loss_history,
            train_accuracy_history,
            validation_accuracy_history,
        ) = ([], [], [], [])

        if train_verb:
            system.freeze_feature_extractors()
        key = "noun_class"
        torch.cuda.empty_cache()
        epoch_start = 0
        system.load_models_to_device(rank, verb=False)
        noun_model = AttentionModel(system.noun_map).to(rank)
        opt_sd = lr_scheduler_sd = None
        if noun_snapshot_path is not None:
            epoch_start, opt_sd = system.load_snapshot(
                noun_snapshot_path, torch.device(rank), noun_model, "ddp_model"
            )
            LOG.info(
                f"Loaded state dict for models from {noun_snapshot_path}, starting at epoch {epoch_start}"
            )
        ddp_model = DDP(noun_model, device_ids=[rank])
        system.opt = system.get_optimizer(ddp_model)
        if opt_sd is not None:
            system.opt.load_state_dict(opt_sd)
            LOG.info(
                f"Loaded state dict for optimizer from {noun_snapshot_path}, starting at epoch {epoch_start}"
            )
        if rank == 0:
            log_print(
                LOG,
                "---------------- ### PHASE 2: TRAINING NOUNS ### ----------------",
                "info",
            )
        ddp_loop(ddp_model, noun_save_path, key, "training loop (nouns)", epoch_start)
        dist.barrier()
        train_stats = {
            "train_accuracy": train_accuracy_history,
            "train_loss": train_loss_history,
            "val_accuracy": validation_accuracy_history,
            "val_loss": validation_loss_history,
        }
        # Write training stats for analysis
        fname = os.path.join(system.cfg.model.save_path, "train_stats_nouns.pkl")
        if rank == 0:
            write_pickle(train_stats, fname)
            LOG.info("Finished noun training")

    writer.close()
    ddp_shutdown()


def ddp_shutdown():
    destroy_process_group()
    # writer.close()


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

    # data_module = EpicActionRecognitionDataModule(cfg)
    # verb_path, noun_path = create_snapshot_paths(Path(cfg.model.save_path))
    # tb_writer = SummaryWriter(cfg.trainer.tb_runs)
    # if cfg.learning.get("ddp", False):
    #     change_port = "Y"
    #     if cfg.learning.ddp.get("master_port", None) is not None:
    #         change_port = input(
    #             f"Would you like to change the DDP master port from {cfg.learning.ddp.master_port} (Y/N)? "
    #         )
    #     if change_port and change_port.upper() == "Y":
    #         new_port = input("Please enter new port ")
    #         cfg.learning.ddp.master_port = new_port

    #     print(f"Proceeding with port {cfg.learning.ddp.master_port}")
    #     system = DDPRecognitionModule(cfg)
    #     set_env_ddp(str(cfg.learning.ddp.get("nccl_p2p_disable", 1)))
    #     run_ddp(
    #         system,
    #         data_module,
    #         cfg.learning.ddp.master_port,
    #         verb_path,
    #         noun_path,
    #         ddp_train,
    #     )
    # else:
    #     system = EpicActionRecognitionModule(cfg)
    #     device = get_device()
    #     train(device, data_module, system, verb_path, noun_path, tb_writer)
    # LOG.info("Training completed!")
    # close_logger(LOG)
    # sys.exit()  # required to prevent CPU lock (soft bug)


if __name__ == "__main__":
    main()

# trainer.gpus=6 learning.ddp.nprocs=6 learning.lr=0.1 learning.batch_size=2 learning.val_batch_size=8
