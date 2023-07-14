from pathlib import Path
import sys
import os
from omegaconf import DictConfig
import hydra
# import torch
# import torch.distributed as dist

# from torch.distributed import destroy_process_group, init_process_group

from systems.recognition_module import EpicActionRecognitionModule, DDPRecognitionModule
from systems.data_module import EpicActionRecognitionDataModule
from utils import get_device, get_loggers


@hydra.main(config_path="../configs", config_name="pilot_config", version_base=None)
def main(cfg: DictConfig):
    data_module = EpicActionRecognitionDataModule(cfg)
    # debug(data_module.val_loader)
    if cfg.learning.get("ddp", False):
        system = DDPRecognitionModule(cfg)
    else:
        system = EpicActionRecognitionModule(cfg)
   
    system.run_training_loop(data_module, cfg.trainer.max_epochs, Path(cfg.model.save_path))
    sys.exit()  # required to prevent CPU lock (soft bug)

def debug(loader):
    for item in loader:
        rgb, flow = item
        rgb_tensors, rgb_metadata = rgb
        print(len(item))
        print(rgb_tensors.size())
        print(rgb_metadata.keys())
        break

def run_ddp(system):
    world_size = system.cfg.trainer.gpus
    batch_size = system.cfg.learning.batch_size
    num_epochs = system.cfg.trainer.max_epochs

    mp.spawn(
        ddp_train,
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
        nprocs=system.cfg.learning.ddp.nprocs,
        join=True,
    )

    writer.close()


 def ddp_setup(self, rank, world_size, master_port):
        # log_print(
        #     LOG,
        #     f"device = {rank}, backend = {self.cfg.learning.ddp.backend}, type = {type(self.cfg.learning.ddp.backend)}, port = {master_port}",
        # )
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        init_process_group(
            self.cfg.learning.ddp.backend, rank=rank, world_size=world_size
        )
        # torch.cuda.set_device(rank)
        LOG.info("Created DDP process group")

def ddp_train(
        rank,
        world_size,
        free_port,
        datamodule,
        system
        verb_save_path,
        noun_save_path,
        batch_size,
        num_epochs,
    ):
    ddp_setup(rank, world_size, free_port)

    train_loader = datamodule.train_dataloader(rank)
    val_loader = datamodule.val_dataloader(rank)
    log_every_n_steps = self.cfg.trainer.get("log_every_n_steps", 1)
    steps_per_run = len(train_loader)
    (
        train_loss_history,
        validation_loss_history,
        train_accuracy_history,
        validation_accuracy_history,
    ) = ([], [], [], [])

    def ddp_loop(ddp_model, snapshot_save_path, key, tqdm_desc="training loop"):
        for epoch in tqdm(range(num_epochs), desc=tqdm_desc, position=0):
            train_loader.sampler.set_epoch(epoch)
            train_loss_meter = ActionMeter("train loss")
            train_acc_meter = ActionMeter("train accuracy")
            for batch in tqdm(
                train_loader,
                desc="train_loader",
                total=len(train_loader),
                position=0,
            ):
                batch_acc, batch_loss = self._step(batch, ddp_model, key)
                train_acc_meter.update(batch_acc, batch_size)
                train_loss_meter.update(batch_loss.item(), batch_size)
                self.backprop(ddp_model, batch_loss)

            train_loss = train_loss_meter.get_average_values()
            train_acc = train_acc_meter.get_average_values()
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_acc)

            # Validation
            if (epoch + 1) % log_every_n_steps == 0:
                val_loader.sampler.set_epoch(epoch)
                ddp_model.eval()
                self.rgb_model.eval()
                self.flow_model.eval()
                val_loss_meter = ActionMeter("val loss")
                val_acc_meter = ActionMeter("val accuracy")
                with torch.no_grad():
                    for batch in tqdm(loader, desc="val_loader", total=len(loader)):  # type: ignore
                        batch_acc, batch_loss = self._step(batch, ddp_model, key)
                        val_acc_meter.update(batch_acc, batch_size)
                        val_loss_meter.update(batch_loss.item(), batch_size)

                val_loss = val_loss_meter.get_average_values()
                val_acc = val_acc_meter.get_average_values()
                validation_loss_history.append(val_loss)
                validation_accuracy_history.append(val_acc)
                if rank == 0:
                    log_print(
                        LOG,
                        f"Epoch:{epoch + 1}"
                        + f" Train Loss: {train_loss}"
                        + f" Val Loss: {val_loss}"
                        + f" Train Accuracy: {train_acc:4f}"
                        + f" Validation Accuracy: {val_acc:.4f}",
                        "info",
                    )
                    writer.add_scalars(
                        tqdm_desc + ": loss",
                        {"train loss": train_loss, "val loss": val_loss},
                        steps_per_run * (epoch + 1),
                    )
                    writer.add_scalars(
                        tqdm_desc + ": accuracy",
                        {"train accuracy": train_acc, "val accuracy": val_acc},
                        steps_per_run * (epoch + 1),
                    )
                    self.save_model(ddp_model, epoch + 1, snapshot_save_path)
                    LOG.info(
                        f"Saved model state for epoch {epoch + 1} at {snapshot_save_path}/checkpoint_{epoch + 1}.pt"
                    )
                if self.early_stopping(val_loss, val_acc):
                    break
            ddp_model.train()
            self.rgb_model.train()
            self.flow_model.train()

    # VERB TRAINING
    self.load_models_to_device(device=rank, verb=True)

    (
        train_loss_history,
        validation_loss_history,
        train_accuracy_history,
        validation_accuracy_history,
    ) = ([], [], [], [])
    ddp_model = DDP(self.verb_model, device_ids=[rank])
    if rank == 0:
        log_print(
            LOG,
            "---------------- ### PHASE 1: TRAINING VERBS ### ----------------",
            "info",
        )
    key = "verb_class"
    self.opt = self.get_optimizer(key)
    ddp_loop(ddp_model, verb_save_path, key, "training loop (verbs)")
    dist.barrier()
    train_stats = {
        "train_accuracy": train_accuracy_history,
        "train_loss": train_loss_history,
        "val_accuracy": validation_accuracy_history,
        "val_loss": validation_loss_history,
    }

    # Write training stats for analysis
    fname = os.path.join(self.cfg.model.save_path, "train_stats_verbs.pkl")
    if rank == 0:
        write_pickle(train_stats, fname)
        LOG.info("Finished verb training")

    # NOUN TRAINING
    self.load_models_to_device(rank, verb=True)
    (
        train_loss_history,
        validation_loss_history,
        train_accuracy_history,
        validation_accuracy_history,
    ) = ([], [], [], [])

    self.freeze_feature_extractors()
    key = "noun_class"
    torch.cuda.empty_cache()
    self.load_models_to_device(rank, verb=False)
    self.opt = self.get_optimizer(key)
    ddp_model = DDP(self.noun_model, device_ids=[rank])
    if rank == 0:
        log_print(
            LOG,
            "---------------- ### PHASE 2: TRAINING NOUNS ### ----------------",
            "info",
        )
    ddp_loop(ddp_model, noun_save_path, key, "training loop (nouns)")
    dist.barrier()
    train_stats = {
        "train_accuracy": train_accuracy_history,
        "train_loss": train_loss_history,
        "val_accuracy": validation_accuracy_history,
        "val_loss": validation_loss_history,
    }
    # Write training stats for analysis
    fname = os.path.join(self.cfg.model.save_path, "train_stats_nouns.pkl")
    if rank == 0:
        write_pickle(train_stats, fname)
        LOG.info("Finished noun training")

    ddp_shutdown()

def ddp_shutdown():
    destroy_process_group()


if __name__ == "__main__":
    main()
