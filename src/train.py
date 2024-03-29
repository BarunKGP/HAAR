from pathlib import Path
import sys
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from models.models import AttentionModel
from systems.recognition_module import EpicActionRecognitionModule, DDPRecognitionModule
from systems.data_module import EpicActionRecognitionDataModule
from utils import (
    ActionMeter,
    get_device,
    get_loggers,
    log_print,
    write_pickle,
    close_logger,
)
from constants import TRAIN_LOGNAME


LOG = get_loggers(__name__, filename=TRAIN_LOGNAME)
# writer = SummaryWriter("data/pilot-01/runs_2")


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
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    # debugging
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


def create_snapshot_paths(model_save_path):
    verb_save_path = model_save_path / "verbs"
    noun_save_path = model_save_path / "nouns"
    verb_save_path.mkdir(parents=True, exist_ok=True)
    noun_save_path.mkdir(parents=True, exist_ok=True)
    LOG.info("Created snapshot paths for verbs and nouns")
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
        LOG.info(f'train loss meter: {train_loss_meter.get_average_values()}')
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
            log_print(LOG, f'Best val_acc = {best_val_acc}')
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
                print(f'val_loss on rank {rank} = {val_loss_meter.get_average_values()}')
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


@hydra.main(config_path="../configs", config_name="pilot_config", version_base=None)
def main(cfg: DictConfig):
    LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
    data_module = EpicActionRecognitionDataModule(cfg)
    verb_path, noun_path = create_snapshot_paths(Path(cfg.model.save_path))
    tb_writer = SummaryWriter(cfg.trainer.tb_runs)
    if cfg.learning.get("ddp", False):
        change_port = "Y"
        if cfg.learning.ddp.get("master_port", None) is not None:
            change_port = input(
                f"Would you like to change the DDP master port from {cfg.learning.ddp.master_port} (Y/N)? "
            )
        if change_port and change_port.upper() == "Y":
            new_port = input("Please enter new port ")
            cfg.learning.ddp.master_port = new_port
        
        print(f"Proceeding with port {cfg.learning.ddp.master_port}")
        system = DDPRecognitionModule(cfg)
        set_env_ddp(str(cfg.learning.ddp.get("nccl_p2p_disable", 1)))
        run_ddp(
            system,
            data_module,
            cfg.learning.ddp.master_port,
            verb_path,
            noun_path,
            ddp_train,
        )
    else:
        system = EpicActionRecognitionModule(cfg)
        device = get_device()
        train(device, data_module, system, verb_path, noun_path, tb_writer)
    LOG.info("Training completed!")
    close_logger(LOG)
    sys.exit()  # required to prevent CPU lock (soft bug)


if __name__ == "__main__":
    main()

# trainer.gpus=6 learning.ddp.nprocs=6 learning.lr=0.1 learning.batch_size=2 learning.val_batch_size=8
