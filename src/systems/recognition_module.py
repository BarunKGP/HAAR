import os
import re
from omegaconf import DictConfig

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import SGD, Adam
from constants import (
    NUM_NOUNS,
    NUM_VERBS,
    TRAIN_PICKLE,
    TEST_PICKLE,
    BATCH_SIZE,
    PICKLE_ROOT,
)
from models.tsm import TSM
from models.models import AttentionModel, WordEmbeddings
from tqdm import tqdm
from frame_loader import FrameLoader
from systems.data_module import EpicActionRecognitionDataModule
from utils import ActionMeter, get_device, get_loggers, write_pickle
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

LOG = get_loggers(name=__name__, filename="data/pilot-01/logs/train.log")


def get_word_map(file_loc):
    try:
        df = pd.read_csv(file_loc)
    except:
        LOG.error(f"invalid file location: {file_loc}", exc_info=True)
        raise FileNotFoundError(f"Invalid file location: {file_loc}")
    return df[["id", "key"]]


def strip_model_prefix(state_dict):
    return {re.sub("^model.", "", k): v for k, v in state_dict.items()}


def load_model(cfg: DictConfig, modality: str, output_dim: int = 0):
    # output_dim: int = sum([class_count for _, class_count in TASK_CLASS_COUNTS])
    LOG.debug("Assigning model state...")
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
        model = WordEmbeddings()
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
    def __init__(self, cfg: DictConfig, datamodule: EpicActionRecognitionDataModule):
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.ddp = self.cfg.learning.get("ddp", False)
        if self.ddp:
            rank = self.device = int(os.environ["LOCAL_RANK"])
        else:
            self.device = get_device()
            rank = None
        print(f"self.device = {self.device}")
        self.train_loader = datamodule.train_dataloader(rank=rank)
        self.val_loader = datamodule.val_dataloader(rank=rank)
        self.test_loader = datamodule.test_dataloader(rank=rank)

        self.narration_model = load_model(self.cfg, modality="narration")
        self.verb_map = get_word_map(self.cfg.data.verb_loc)
        self.noun_map = get_word_map(self.cfg.data.noun_loc)
        self.verb_embeddings = self.get_embeddings("verb")
        self.noun_embeddings = self.get_embeddings("noun")
        self.verb_one_hot = F.one_hot(torch.arange(0, NUM_VERBS))
        self.noun_one_hot = F.one_hot(torch.arange(0, NUM_NOUNS))

        self.verb_model = AttentionModel(self.verb_embeddings, self.verb_map)
        self.noun_model = AttentionModel(self.noun_embeddings, self.noun_map)
        self.rgb_model = load_model(self.cfg, modality="rgb")
        self.flow_model = load_model(self.cfg, modality="flow")

        # self.debug()
        self.opt = self.get_optimizer()

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

    def debug(self):
        for param in self.rgb_model.parameters():
            if not param.requires_grad:
                print(param)

        print(torch.cuda.memory_summary())

    def get_embeddings(self, mode):
        if mode == "verb":
            text = self.verb_map["key"].values.tolist()
        elif mode == "noun":
            text = self.noun_map["key"].values.tolist()
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')
        embeddings = self.narration_model(text)
        return embeddings

    def get_model(self):
        return self.attention_model

    def get_optimizer(self):
        if self.ddp:
            att_model_verb = self.verb_model.module
            att_model_noun = self.noun_model.module
        else:
            att_model_verb = self.verb_model
            att_model_noun = self.noun_model
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
                        {"params": att_model_verb.parameters()},
                        {"params": att_model_noun.parameters()},
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
                        {"params": att_model_verb.parameters()},
                        {"params": att_model_noun.parameters()},
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
                {"params": att_model_verb.parameters()},
                {"params": att_model_noun.parameters()},
            ],
            lr=lr,
            momentum=momentum,
        )

    def save_model(self, epoch) -> None:
        """
        Saves the model state and optimizer state on the dict

        __args__:
            epoch (int): number of epochs the model has been
                trained for
        """
        path = self.cfg.model.get(
            "save_path", None
        )  #! figure out correct config mapping
        if path is None:
            path = r"./checkpoints"
        torch.save(
            {
                "epoch": epoch,
                "rgb_model": self.rgb_model.cpu().state_dict(),
                "flow_model_state_dict": self.flow_model.cpu().state_dict(),
                "attention_model_state_dict": self.attention_model.module().state_dict() if self.ddp else self.attention_model.state_dict(),  # type: ignore
                "optimizer_state_dict": self.opt.state_dict(),
            },
            os.path.join(path, f"checkpoint_{epoch}.pt"),
        )

    def _train(self, loader, key):
        """Run the training loop for one epoch.
        Calculate and return the loss and accuracy.
        Separate loop for val/test - no backprop.

        Returns:
            (float, float, float, float): average loss and accuracy
        """
        # self.attention_model.train()
        train_loss_meter = ActionMeter("train loss")
        train_acc_meter = ActionMeter("train accuracy")

        for batch in tqdm(loader, desc="train_loader", total=len(loader)):
            batch_acc, batch_loss = self._step(batch, key)
            train_acc_meter.update(batch_acc, len(batch[0]))
            train_loss_meter.update(batch_loss.item(), len(batch[0]))

            # Backpropagate and optimize
            self.attention_model.zero_grad()
            batch_loss.backward()
            clip_grad_norm_(
                self.attention_model.parameters(), self.cfg.trainer.gradient_clip_val
            )
            self.opt.step()
        return (
            train_loss_meter.avg_verb,
            train_loss_meter.avg_noun,
            train_acc_meter.avg_verb,
            train_acc_meter.avg_noun,
        )

    def _validate(self, loader, key):
        self.attention_model.eval()
        val_loss_meter = ActionMeter("val loss")
        val_acc_meter = ActionMeter("val accuracy")

        for batch in tqdm(loader, desc="val_loader", total=len(loader)):  # type: ignore
            n = batch[0].shape[0]  # batch size
            batch_acc, batch_loss = self._step(batch, key)
            val_acc_meter.update(batch_acc, n)
            val_loss_meter.update(batch_loss.item(), n)

        self.attention_model.train()
        return (
            val_loss_meter.avg_verb,
            val_loss_meter.avg_noun,
            val_acc_meter.avg_verb,
            val_acc_meter.avg_noun,
        )

    def _step(self, batch, key):
        """One step of the optimization process. This
        method is run in all of train/val/test
        """
        assert key in ["verb_class", "noun_class"], "invalid key"
        rgb, flow = batch
        rgb_images, metadata = rgb  # rgb and flow metadata are the same
        flow_images = flow[0]
        word_class = metadata[key]
        text = metadata["narration"]
        rgb_feats = self.rgb_model(rgb_images.to(self.device))
        flow_feats = self.flow_model(flow_images.to(self.device))
        narration_feats = self.narration_model(text)
        feats = torch.hstack((rgb_feats, flow_feats, narration_feats.to(self.device)))
        #! Following should be handled by DistributedSampler
        # feats = feats.to(self.device)
        # verb_class = verb_class.to(self.device)
        # noun_class = noun_class.to(self.device)
        #! Should use DDP model here
        if key == "verb_class":
            predictions = self.verb_model(feats, word_class)
        else:
            predictions = self.noun_model(feats, word_class)
        predictions = predictions.detach()
        batch_acc = self.compute_accuracy(predictions, word_class)
        batch_loss = self.compute_loss(predictions, word_class)

        print(f"accuracy types: batch_loss: {type(batch_loss)}, batch_acc: {batch_acc}")
        return batch_acc, batch_loss

    def compute_accuracy(self, preds, labels):
        print(f"preds device = {preds.device()}, labels device = {labels.device}")
        with torch.no_grad():
            preds = torch.argmax(preds, dim=1)
            correct = (preds == labels.cpu()).float().sum().item()
            batch_accuracy = correct / len(preds)
        return batch_accuracy

    def compute_loss(self, preds, labels, is_normalize=True):
        loss = self.loss_fn(preds, labels)
        if is_normalize:
            loss = loss / len(preds)
        return loss

    def load_models_to_device(self, train=True):
        self.rgb_model = self.rgb_model.to(self.device)
        self.flow_model = self.flow_model.to(self.device)
        self.narration_model = self.narration_model.to(self.device)
        self.verb_model = self.verb_model.to(self.device)
        # self.noun_model = self.noun_model.to(self.device)
        if self.cfg.learning.get("ddp", False):
            self.attention_model = DDP(self.attention_model, device_ids=[self.device])  # type: ignore
        if train:
            self.rgb_model.train()
            self.flow_model.train()
            self.verb_model.train()
            self.noun_model.train()

    # baseline paper used num_epochs = 3e6
    def training_loop(self, num_epochs=500, model_save_path=None):
        """Run the main training loop for the model.
        May need to run separate loops for train/test.

        Args:
            num_epochs (int, optional): number of training epochs.
                Defaults to 500.
            model_save_path (str, optional): path to save model during
                and after training. Defaults to self.save_path
        """
        if model_save_path is None:
            model_save_path = self.cfg.save_path  # ? configure a default save_path?

        self.load_models_to_device()
        for epoch in tqdm(range(num_epochs), desc="training_verbs"):
            if self.ddp:
                self.train_loader.sampler.set_epoch(epoch)
            train_loss_verb, _, train_acc_verb, _ = self._train(
                self.train_loader, "verb_class"
            )
            self.train_loss_history.append(train_loss_verb)
            self.train_accuracy_history.append(train_acc_verb)

            if epoch % 10 == 0:
                if self.ddp:
                    self.val_loader.sampler.set_epoch(epoch)
                val_loss_verb, _, val_acc_verb, _ = self._validate(
                    self.val_loader, "verb_class"
                )
                self.validation_loss_history.append(val_loss_verb)
                self.validation_accuracy_history.append(val_acc_verb)
                LOG.info(
                    f"Epoch:{epoch + 1}"
                    + f" Train Loss: {train_loss_verb}"
                    + f" Val Loss: {val_loss_verb}"
                    + f" Train Accuracy: {train_acc_verb:4f}"
                    + f" Validation Accuracy: {val_acc_verb:.4f}"
                )

        # Write training stats for analysis
        train_stats = {
            "accuracy": self.train_accuracy_history,
            "loss": self.train_loss_history,
        }
        LOG.info("Finished training")
        fname = os.path.join(model_save_path, "train_stats.pkl")
        write_pickle(train_stats, fname)
        self.save_model(num_epochs)
