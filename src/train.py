import gc

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import (
    BATCH_SIZE,
    NOUN_CLASSES,
    NUM_NOUNS,
    NUM_VERBS,
    PICKLE_ROOT,
    TEST_PICKLE,
    TRAIN_PICKLE,
    VERB_CLASSES,
)
from frame_loader import FrameLoader
from models import AttentionModel, WordEmbeddings
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ActionMeter, get_device, get_loggers, write_pickle

# LOGGING
stream_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = RotatingFileHandler(
    filename="logs/train.log", maxBytes=50000, backupCount=5
)
logger = get_loggers(
    name=__name__,
    handlers=[(stream_handler, logging.INFO), (file_handler, logging.ERROR)],
)


def get_dataloader(train=True):
    if train:
        dataset = FrameLoader(
            # loc = os.path.join(PICKLE_ROOT, 'samples/df_train100_first10.pkl'),
            loc=TRAIN_PICKLE,
            info_loc=os.path.join(PICKLE_ROOT, "video_info.pkl"),
        )
    else:
        dataset = FrameLoader(
            # loc = os.path.join(PICKLE_ROOT, 'samples/df_train100_first10.pkl'),
            loc=TEST_PICKLE,
            info_loc=os.path.join(PICKLE_ROOT, "video_info.pkl"),
        )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


def get_word_map(file_loc):
    try:
        df = pd.read_csv(file_loc)
    except:
        logger.error(f"Invalid file location: {file_loc}", exc_info=True)
        raise FileNotFoundError(f"Invalid file location: {file_loc}")
    return df[["id", "key"]]


class Trainer(object):
    def __init__(self, verb_loc, noun_loc, loss_fn, save_path="data/train_run") -> None:
        self.loss_fn = loss_fn
        self.save_path = save_path

        self.train_loader = get_dataloader()
        self.val_loader = get_dataloader(train=False)
        self.device = get_device()
        self.verb_map = get_word_map(verb_loc)
        self.noun_map = get_word_map(noun_loc)

        self.embedding_model = WordEmbeddings()

        self.verb_embeddings = self.get_embeddings("verb")
        self.noun_embeddings = self.get_embeddings("noun")
        self.verb_one_hot = F.one_hot(torch.arange(0, NUM_VERBS))
        self.noun_one_hot = F.one_hot(torch.arange(0, NUM_NOUNS))

        self.attention_model = AttentionModel(
            self.verb_embeddings, self.noun_embeddings, self.verb_map, self.noun_map
        )
        self.opt = torch.optim.Adam(
            self.attention_model.parameters(), lr=1e-5, weight_decay=1e-5
        )

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

    def get_embeddings(self, mode):
        if mode == "verb":
            text = self.verb_map["key"].values.tolist()
        elif mode == "noun":
            text = self.noun_map["key"].values.tolist()
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')
        embeddings = self.embedding_model(text)
        return embeddings

    def get_model(self):
        return self.attention_model

    def save_model(self, epoch, path=None) -> None:
        """
        Saves the model state and optimizer state on the dict

        __args__:
            epoch (int): number of epochs the model has been
                trained for
            path [Optional(str)]: path where to save the model.
                Defaults to self.save_path
        """
        if path is None:
            path = self.save_path
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.attention_model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            os.path.join(path, f"checkpoint_{epoch}.pt"),
        )

    def _train(self):
        """Run the training loop for one epoch.
        Calculate and return the loss and accuracy.
        Separate loop for val/test - no backprop.

        Returns:
            (float, float, float, float): average loss and accuracy
        """
        self.attention_model.train()
        loader = self.train_loader
        train_loss_meter = ActionMeter("train loss")
        train_acc_meter = ActionMeter("train accuracy")

        # loop over each minibatch
        for feats, verb_class, noun_class in tqdm(
            loader, desc="train_loader", total=len(loader)
        ):
            feats = feats.to(self.device)
            verb_class = verb_class.to(self.device)
            noun_class = noun_class.to(self.device)
            n = feats.shape[0]  # batch_size

            predictions_verb, predictions_noun = self.attention_model(
                feats, verb_class, noun_class
            )

            batch_acc_noun = self.compute_accuracy(predictions_noun, noun_class)
            batch_acc_verb = self.compute_accuracy(predictions_verb, verb_class)

            # * Pytorch multi-loss reference: https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            batch_loss_verb = self.compute_loss(predictions_verb, verb_class)
            batch_loss_noun = self.compute_loss(predictions_noun, noun_class)
            batch_loss = batch_loss_noun + batch_loss_verb

            train_acc_meter.update(batch_acc_verb, batch_acc_noun, n)
            train_loss_meter.update(batch_loss_verb.item(), batch_loss_noun.item(), n)

            # Backpropagate and optimize
            self.attention_model.zero_grad()
            batch_loss.backward()
            self.opt.step()
        return (
            train_loss_meter.avg_verb,
            train_loss_meter.avg_noun,
            train_acc_meter.avg_verb,
            train_acc_meter.avg_noun,
        )

    def _validate(self):
        self.attention_model.eval()
        loader = self.val_loader
        val_loss_meter = ActionMeter("val loss")
        val_acc_meter = ActionMeter("val accuracy")

        # loop over each minibatch
        for feats, verb_class, noun_class in tqdm(
            loader, desc="val_loader", total=len(loader)
        ):
            feats = feats.to(self.device)
            verb_class = verb_class.to(self.device)
            noun_class = noun_class.to(self.device)
            n = feats.shape[0]

            predictions_verb, predictions_noun = self.attention_model(
                feats, verb_class, noun_class
            )

            batch_acc_noun = self.compute_accuracy(predictions_noun, noun_class)
            batch_acc_verb = self.compute_accuracy(predictions_verb, verb_class)

            batch_loss_verb = self.compute_loss(predictions_verb, verb_class)
            batch_loss_noun = self.compute_loss(predictions_noun, noun_class)

            val_acc_meter.update(batch_acc_verb, batch_acc_noun, n)
            val_loss_meter.update(batch_loss_verb.item(), batch_loss_noun.item(), n)
        return (
            val_loss_meter.avg_verb,
            val_loss_meter.avg_noun,
            val_acc_meter.avg_verb,
            val_acc_meter.avg_noun,
        )

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
            model_save_path = self.save_path
        self.attention_model.to(self.device)
        for epoch in tqdm(range(num_epochs), desc="epoch"):
            loss_verb, loss_noun, acc_verb, acc_noun = self._train()
            train_loss = loss_noun + loss_verb
            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append((acc_verb, acc_noun))

            if epoch % 100 == 0:
                (
                    val_loss_verb,
                    val_loss_noun,
                    val_verb_acc,
                    val_noun_acc,
                ) = self._validate()
                val_loss = val_loss_noun + val_loss_verb
                self.validation_loss_history.append(val_loss)
                self.validation_accuracy_history.append((val_verb_acc, val_noun_acc))
                logger.info(
                    f"Epoch:{epoch + 1}"
                    + f" Train Loss (verb/noun):{loss_verb}/{loss_noun}"
                    + f" Val Loss (verb/noun): {val_loss_verb}/{val_loss_noun}"
                    + f" Train Accuracy (verb/noun): {acc_verb:.4f}/{acc_noun:.4f}"
                    + f" Validation Accuracy (verb/noun): {val_verb_acc:.4f}/{val_noun_acc:.4f}"
                )
                self.save_model(epoch, path=model_save_path)
            # print('GPU usage:')
            # print(torch.cuda.list_gpu_processes(self.device))
        # gc.collect()
        # torch.cuda.empty_cache()

        # Write training stats for analysis
        train_stats = {
            "accuracy": self.train_accuracy_history,
            "loss": self.train_loss_history,
        }
        logger.log(level=35, msg="Finished training")
        fname = os.path.join(model_save_path, "train_stats.pkl")
        write_pickle(train_stats, fname)


if __name__ == "__main__":
    trainer = Trainer(
        VERB_CLASSES,
        NOUN_CLASSES,
        nn.CrossEntropyLoss(reduction="mean"),
        save_path="data/pilot-01",
    )
    trainer.training_loop(num_epochs=1)
