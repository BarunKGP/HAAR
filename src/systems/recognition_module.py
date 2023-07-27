# sys.path.append(sys.path[0] + "/..")

import sys
import os
from pathlib import Path

# from contextlib import closing
import socket
import re
from omegaconf import DictConfig, OmegaConf
from systems.data_module import EpicActionRecognitionDataModule

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from constants import (
    NUM_NOUNS,
    NUM_VERBS,
)
from models.tsm import TSM
from models.models import AttentionModel, WordEmbeddings
from tqdm import tqdm
from frame_loader import FrameLoader
from systems.data_module import EpicActionRecognitionDataModule
from utils import ActionMeter, get_device, get_loggers, write_pickle, log_print

LOG = get_loggers(name=__name__, filename="data/pilot-01/logs/recog.log")
writer = SummaryWriter("data/pilot-01/runs_2")


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s.settimeout(30)
    s.bind(("", 0))
    open_port = s.getsockname()[1]
    s.close()
    return open_port


def get_word_map(file_loc):
    try:
        df = pd.read_csv(file_loc)
    except:
        LOG.error(f"invalid file location: {file_loc}", exc_info=True)
        raise FileNotFoundError(f"Invalid file location: {file_loc}")
    return df[["id", "key"]]


def strip_model_prefix(state_dict):
    return {re.sub("^model.", "", k): v for k, v in state_dict.items()}


def load_model(cfg: DictConfig, modality: str, output_dim: int = 0, device="cpu"):
    # output_dim: int = sum([class_count for _, class_count in TASK_CLASS_COUNTS])
    # LOG.info(f'device = {device}')
    if modality in ["rgb", "flow"]:
        if cfg.model.type == "TSM":  # type: ignore
            model = TSM(
                num_class=output_dim if output_dim > 0 else cfg.model.num_class,
                num_segments=cfg.data.frame_count,
                modality=modality,
                base_model=cfg.model.backbone,
                segment_length=cfg["data"][modality]["segment_length"],
                consensus_type="avg",
                dropout=cfg.model.dropout,
                partial_bn=cfg.model.partial_bn,
                pretrained=cfg.model.get("pretrained", None),
                shift_div=cfg.model.shift_div,
                non_local=cfg.model.non_local,
                temporal_pool=cfg.model.temporal_pool,
                freeze_train_layers=cfg.model.get("use_pretrained", True),
            )
        else:
            raise ValueError(f"Unknown model type {cfg.model.type!r}")

        if cfg.model.get("weights", None) is not None:
            if cfg.model.pretrained is not None:
                LOG.warning(
                    f"model.pretrained was set to {cfg.model.pretrained!r} but "
                    f"you also specified to load weights from {cfg.model.weights}."
                    "The latter will take precedence."
                )
            weight_loc = cfg.model.weights[modality]
            LOG.info(f"Loading weights from {weight_loc}")
            state_dict = torch.load(weight_loc, map_location=torch.device("cpu"))
            if "state_dict" in state_dict:
                # Person is trying to load a checkpoint with a state_dict key, so we pull
                # that out.
                LOG.info("Stripping 'model' prefix from pretrained state_dict keys")
                sd = strip_model_prefix(state_dict["state_dict"])
                # Change shape of final linear layer
                sd["new_fc.weight"] = torch.rand([1024, 2048], requires_grad=True)
                sd["new_fc.bias"] = torch.rand(1024, requires_grad=True)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if len(missing) > 0:
                    LOG.warning(f"Missing keys in checkpoint: {missing}")
                if len(unexpected) > 0:
                    LOG.warning(f"Unexpected keys in checkpoint: {unexpected}")
    elif modality == "narration":
        if type(device) == int:
            device = "cuda:" + str(device)
        model = WordEmbeddings(device=device)
        narr_cfg = cfg.model.get("narration_model", None)
        if narr_cfg and narr_cfg.get("pretrained", False):
            for param in model.parameters():
                param.requires_grad = False
    else:
        LOG.error(
            f"Incorrect modality {modality} passed. Modalities must be among ['flow', 'rgb', 'narration']."
        )
        raise Exception("Incorrect modalities specified")
    return model


class EpicActionRecognitionModule(object):
    def __init__(self, cfg: DictConfig, device=None):
        self.cfg = cfg
        LOG.info("Config:\n" + OmegaConf.to_yaml(cfg))
        self.loss_fn = nn.CrossEntropyLoss()
        self.verb_map = get_word_map(self.cfg.data.verb_loc)
        self.noun_map = get_word_map(self.cfg.data.noun_loc)
        self.verb_one_hot = F.one_hot(torch.arange(0, NUM_VERBS))
        self.noun_one_hot = F.one_hot(torch.arange(0, NUM_NOUNS))

        self.verb_embeddings = None
        self.noun_embeddings = None
        self.rgb_model = None
        self.flow_model = None
        self.narration_model = None
        # self.verb_model = None
        # self.noun_model = None
        self.opt = None

        self.ddp = self.cfg.learning.get("ddp", False)
        if self.ddp:
            self.device = None
        else:
            self.device = device if device is not None else get_device()

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

    def debug(self, model):
        for param in model.parameters():
            if not param.requires_grad:
                print(param)

        print(torch.cuda.memory_summary())

    def get_embeddings(self, key):
        assert (
            self.narration_model is not None
        ), "Narration model has not been initialized"
        if key.lower() == "verb":
            text = self.verb_map["key"].values.tolist()
        elif key.lower() == "noun":
            text = self.noun_map["key"].values.tolist()
        else:
            raise Exception('Invalid key: choose either "noun" or "verb"')
        embeddings = self.narration_model(text)
        return embeddings

    def get_model(self, key):
        if key.lower() == "verb":
            return self.verb_model
        elif key.lower() == "noun":
            return self.noun_model
        else:
            raise Exception('Invalid key: choose either "noun" or "verb"')

    def get_optimizer(self, model):
        assert self.rgb_model is not None, "RGB model has not been initialized"
        assert self.flow_model is not None, "FLow model has not been initialized"

        if "optimizer" in self.cfg.learning:
            cfg = self.cfg.learning.optimizer
            if cfg["type"] == "Adam":
                return Adam(
                    [
                        {
                            "params": filter(
                                lambda p: p.requires_grad, self.rgb_model.parameters()
                            ),
                        },
                        {
                            "params": filter(
                                lambda p: p.requires_grad, self.flow_model.parameters()
                            ),
                        },
                        {"params": model.parameters()},
                        # {"params": self.noun_model.parameters()},
                    ],
                    lr=cfg.lr,
                )
            elif cfg["type"] == "SGD":
                return SGD(
                    [
                        {
                            "params": filter(
                                lambda p: p.requires_grad, self.rgb_model.parameters()
                            ),
                        },
                        {
                            "params": filter(
                                lambda p: p.requires_grad, self.flow_model.parameters()
                            ),
                        },
                        {"params": model.parameters()},
                        # {"params": self.noun_model.parameters()},
                    ],
                    lr=self.cfg.learning.lr,
                    momentum=cfg.momentum,
                )
            else:
                LOG.error(
                    "Incorrect optimizer chosen. Proceeding with SGD optimizer as default"
                )

        # Default optimizer
        lr = 0.01
        momentum = 0.9
        return SGD(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.rgb_model.parameters()
                    ),
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.flow_model.parameters()
                    ),
                },
                {"params": model.parameters()},
                # {"params": self.noun_model.parameters()},
            ],
            lr=lr,
            momentum=momentum,
        )

    def save_model(self, model, epoch, path) -> None:
        """
        Saves the model state and optimizer state on the dict

        __args__:
            epoch (int): number of epochs the model has been
                trained for
            path (Path): path to save the model
        """
        assert self.rgb_model is not None, "RGB model has not been initialized"
        assert self.flow_model is not None, "FLow model has not been initialized"
        assert (
            self.narration_model is not None
        ), "Narration model has not been initialized"
        assert self.opt is not None, "Optimizer has not been initialized"
        torch.save(
            {
                "epoch": epoch,
                "rgb_model": self.rgb_model.state_dict(),
                "flow_model": self.flow_model.state_dict(),
                "attention_model": model.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            os.path.join(path, f"checkpoint_{epoch}.pt"),
        )

    def load_snapshot(self, snapshot_path, device, attention_model, model_key="attention_model"):
        checkpoint = torch.load(snapshot_path, map_location=device)
        epoch = checkpoint['epoch']
        self.rgb_model.load_state_dict(checkpoint['rgb_model'])
        self.flow_model.load_state_dict(checkpoint['flow_model'])
        attention_model.load_state_dict(checkpoint[model_key])
        opt_sd = checkpoint['optimizer']

        return epoch, opt_sd


    def _train(self, loader, key):
        """Run the training loop for one epoch.
        Calculate and return the loss and accuracy.
        Separate loop for val/test - no backprop.

        Returns:
            (float, float, float, float): average loss and accuracy
        """
        assert key in ["verb_class", "noun_class"], "invalid key"
        train_loss_meter = ActionMeter("train loss")
        train_acc_meter = ActionMeter("train accuracy")
        batch_size = self.cfg.learning.batch_size
        model = verb_model if key == "verb_class" else self.noun_model
        for batch in tqdm(
            loader, desc="train_loader", total=len(loader), position=0, leave=True
        ):
            batch_acc, batch_loss = self._step(batch, model, key)
            train_acc_meter.update(batch_acc, batch_size)
            train_loss_meter.update(batch_loss.item(), batch_size)

            if key == "verb_class":
                self.backprop(self.verb_model, batch_loss)
            else:
                self.backprop(self.noun_model, batch_loss)

        return (
            train_loss_meter.get_average_values(),
            train_acc_meter.get_average_values(),
        )

    def _validate(self, loader, key):
        assert key in ["verb_class", "noun_class"], "invalid key"

        torch.cuda.empty_cache()
        if key == "verb_class":
            assert (
                self.verb_model is not None
            ), "AttentionModel has not been initialized"
            self.verb_model.eval()
        else:
            assert (
                self.noun_model is not None
            ), "AttentionModel has not been initialized"
            self.noun_model.eval()
        model = self.verb_model if key == "verb_class" else self.noun_model
        val_loss_meter = ActionMeter("val loss")
        val_acc_meter = ActionMeter("val accuracy")
        batch_size = self.cfg.learning.batch_size
        with torch.no_grad():
            for batch in tqdm(loader, desc="val_loader", total=len(loader)):  # type: ignore
                batch_acc, batch_loss = self._step(batch, model, key)
                val_acc_meter.update(batch_acc, batch_size)
                val_loss_meter.update(batch_loss.item(), batch_size)

        if key == "verb_class":
            self.verb_model.train()
        else:
            self.noun_model.train()
        return (
            val_loss_meter.get_average_values(),
            val_acc_meter.get_average_values(),
        )

    def _step(self, batch, model, key):
        """One step of the optimization process. This
        method is run in all of train/val/test
        """
        assert self.rgb_model is not None, "RGB model has not been initialized"
        assert self.flow_model is not None, "FLow model has not been initialized"
        assert (
            self.narration_model is not None
        ), "Narration model has not been initialized"
        rgb, flow = batch
        rgb_images, metadata = rgb  # rgb and flow metadata are the same
        flow_images = flow[0]
        labels = metadata[key]
        text = metadata["narration"]
        # Feature extraction
        rgb_feats = self.rgb_model(rgb_images.to(self.device))
        flow_feats = self.flow_model(flow_images.to(self.device))
        narration_feats = self.narration_model(text)
        # LOG.info(f'narration feats on: {narration_feats.device}')
        feats = torch.hstack((rgb_feats, flow_feats, narration_feats.to(self.device)))

        # Predictions
        if key == "verb_class":
            predictions = model(feats, labels, self.verb_embeddings)
        else:
            predictions = model(feats, labels, self.noun_embeddings)
        predictions = predictions.cpu()

        # Compute loss and accuracy
        batch_acc = self.compute_accuracy(predictions, labels)
        batch_loss = self.compute_loss(predictions, labels)

        return batch_acc, batch_loss

    def backprop(self, model, loss):
        assert self.opt is not None, "Optimizer has not been initialized"
        model.zero_grad()
        self.opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), self.cfg.trainer.gradient_clip_val)
        self.opt.step()

    def compute_accuracy(self, preds, labels):
        with torch.no_grad():
            preds = torch.argmax(preds, dim=1)
            correct = (preds == labels).float().sum().item()
            batch_accuracy = correct / len(preds)
        return batch_accuracy

    def compute_loss(self, preds, labels, is_normalize=True):
        loss = self.loss_fn(preds, labels)
        if is_normalize:
            loss = loss / len(preds)
        return loss

    def load_models_to_device(self, device=None, verb=True):
        if device is None:
            device = self.device
        assert device is not None, "device not set"
        self.narration_model = load_model(self.cfg, modality="narration", device=device)
        self.rgb_model = load_model(self.cfg, modality="rgb", device=device)
        self.flow_model = load_model(self.cfg, modality="flow", device=device)

        self.rgb_model.to(device)
        self.flow_model.to(device)
        # self.narration_model.to(device)
        # self.rgb_model.train()
        # self.flow_model.train()

        if verb:
            self.verb_embeddings = self.get_embeddings("verb").to(device)
            # self.verb_model = AttentionModel(
            #     self.verb_embeddings, self.verb_map, device=device
            # ).to(device)
            # self.verb_model.train()
        else:
            self.noun_embeddings = self.get_embeddings("noun").to(device)
            # self.noun_model = AttentionModel(
            #     self.noun_embeddings, self.noun_map, device=device
            # ).to(device)
            # self.noun_model.train()

        LOG.info("Loaded models to device")

    def early_stopping(self, loss, accuracy):
        if self.cfg.trainer.get("early_stopping", False):
            threshold = self.cfg.trainer.early_stopping["threshold"]
            if (
                self.cfg.trainer.early_stopping.criterion == "accuracy"
                and accuracy >= threshold
            ):
                return True
            elif (
                self.cfg.trainer.early_stopping.criterion == "loss"
                and loss <= threshold
            ):
                return True
        return False

    # baseline paper used num_epochs = 3e6

    def run_training_loop(self, datamodule, num_epochs: int = 50, model_save_path=None):
        """Run the main training loop for the model.
        May need to run separate loops for train/test.

        Args:
            num_epochs (int, optional): number of training epochs.
                Defaults to 500.
            model_save_path (str, optional): path to save model during
                and after training. Defaults to self.save_path
        """
       
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        self.load_models_to_device(verb=True)
        verb_model = AttentionModel(self.verb_map).to(self.device)
        verb_save_path = model_save_path / "verbs"
        noun_save_path = model_save_path / "nouns"
        verb_save_path.mkdir(parents=True, exist_ok=True)
        noun_save_path.mkdir(parents=True, exist_ok=True)
        LOG.info("Created snapshot paths for verbs and nouns")
        log_every_n_steps = self.cfg.trainer.get("log_every_n_steps", 1)
        steps_per_run = len(train_loader)
        self.opt = self.get_optimizer(verb_model)
        LOG.info("---------------- ### PHASE 1: TRAINING VERBS ### ----------------")
        for epoch in tqdm(range(num_epochs), desc="training loop(verb)", position=0):
            train_loss_verb, train_acc_verb = self._train(train_loader, "verb_class")
            self.train_loss_history.append(train_loss_verb)
            self.train_accuracy_history.append(train_acc_verb)

            if (epoch + 1) % log_every_n_steps == 0:
                val_loss_verb, val_acc_verb = self._validate(val_loader, "verb_class")
                self.validation_loss_history.append(val_loss_verb)
                self.validation_accuracy_history.append(val_acc_verb)
                LOG.info(
                    f"Epoch:{epoch + 1}"
                    + f" Train Loss: {train_loss_verb}"
                    + f" Val Loss: {val_loss_verb}"
                    + f" Train Accuracy: {train_acc_verb:4f}"
                    + f" Validation Accuracy: {val_acc_verb:.4f}"
                )
                writer.add_scalars(
                    "loss",
                    {"train loss": train_loss_verb, "val loss": val_loss_verb},
                    steps_per_run * (epoch + 1),
                )
                writer.add_scalars(
                    "accuracy",
                    {"train accuracy": train_acc_verb, "val accuracy": val_acc_verb},
                    steps_per_run * (epoch + 1),
                )
                self.save_model(verb_model, epoch + 1, verb_save_path)
                LOG.info(
                    f"Saved model state for epoch {epoch + 1} at {verb_save_path}/checkpoint_{epoch + 1}.pt"
                )
                if self.early_stopping(val_loss_verb, val_acc_verb):
                    break

        # Write training stats for analysis
        train_stats = {
            "train_accuracy": self.train_accuracy_history,
            "train_loss": self.train_loss_history,
            "val_accuracy": self.validation_accuracy_history,
            "val_loss": self.validation_loss_history,
        }
        fname = os.path.join(model_save_path, "train_stats_verbs.pkl")
        write_pickle(train_stats, fname)
        # self.save_model(num_epochs, verb_save_path)
        LOG.info("Finished verb training")

        torch.cuda.empty_cache()
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        self.freeze_feature_extractors()  # We want noun model to use the same params for feature extraction as the verbs
        self.load_models_to_device(verb=False)
        noun_model = AttentionModel(self.noun_map).to(self.device)
        self.opt = self.get_optimizer(noun_model)
        LOG.info("---------------- ### PHASE 2: TRAINING NOUNS ### ----------------")
        for epoch in tqdm(range(num_epochs)):
            train_loss_noun, train_acc_noun = self._train(train_loader, "noun_class")
            self.train_loss_history.append(train_loss_noun)
            self.train_accuracy_history.append(train_acc_noun)

            if (epoch + 1) % log_every_n_steps == 0:
                val_loss_noun, val_acc_noun = self._validate(val_loader, "noun_class")
                self.validation_loss_history.append(val_loss_noun)
                self.validation_accuracy_history.append(val_acc_noun)
                LOG.info(
                    f"Epoch:{epoch + 1}"
                    + f" Train Loss: {train_loss_noun}"
                    + f" Val Loss: {val_loss_noun}"
                    + f" Train Accuracy: {train_acc_noun:4f}"
                    + f" Validation Accuracy: {val_acc_noun:.4f}"
                )
                writer.add_scalars(
                    "loss",
                    {"train loss": train_loss_noun, "val loss": val_loss_noun},
                    steps_per_run * (epoch + 1),
                )
                writer.add_scalars(
                    "accuracy",
                    {"train accuracy": train_acc_noun, "val accuracy": val_acc_noun},
                    steps_per_run * (epoch + 1),
                )
                self.save_model(noun_model, epoch + 1, noun_save_path)
                if self.early_stopping(val_loss_noun, val_acc_noun):
                    break

        train_stats = {
            "train_accuracy": self.train_accuracy_history,
            "train_loss": self.train_loss_history,
            "val_accuracy": self.validation_accuracy_history,
            "val_loss": self.validation_loss_history,
        }
        fname = os.path.join(model_save_path, "train_stats_nouns.pkl")
        write_pickle(train_stats, fname)
        # self.save_model(num_epochs, noun_save_path)
        LOG.info("Finished noun training")

    def freeze_feature_extractors(self):
        for model in [self.rgb_model, self.flow_model, self.narration_model]:
            if model is None:
                continue
            for params in model.parameters():
                params.requires_grad = False
        LOG.info("RGB, Flow and Narration model parameters frozen for training.")


class DDPRecognitionModule(EpicActionRecognitionModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)


    def save_model(self, ddp_model, epoch, path):
        assert self.rgb_model is not None, "RGB model has not been initialized"
        assert self.flow_model is not None, "FLow model has not been initialized"
        torch.save(
            {
                "epoch": epoch,
                "rgb_model": self.rgb_model.state_dict(),
                "flow_model": self.flow_model.state_dict(),
                "ddp_model": ddp_model.module.state_dict(),
                "optimizer": self.opt.state_dict(),
            },
            os.path.join(path, f"checkpoint_{epoch}.pt"),
        )

    # def load_snapshot(self, snapshot_path, model, device):
    #     epoch, new_model = super().load_snapshot(snapshot_path, model, device)
    #     checkpoint = torch.load(snapshot_path, map_location=device)
    #     epoch = checkpoint['epoch']
    #     self.rgb_model.load_state_dict(checkpoint['rgb_model'])
    #     self.flow_model.load_state_dict(checkpoint['flow_model'])
    #     ddp_model.load_state_dict(checkpoint['ddp_model'])
    #     self.opt.load_state_dict(checkpoint['optimizer'])

    #     return epoch, attention_model
