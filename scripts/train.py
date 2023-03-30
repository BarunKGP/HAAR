import os

import torch
from tqdm import tqdm
from frame_loader import FrameLoader
from constants import BATCH_SIZE, PICKLE_ROOT, NUM_NOUNS, NUM_VERBS, VERB_CLASSES, NOUN_CLASSES

import pandas as pd
import pickle

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from models import AttentionModel, WordEmbeddings

from utils import ActionMeter, AverageMeter, get_device


def get_dataloader(train=True):
    print('Creating dataloader...')
    dataset = FrameLoader(
        loc = os.path.join(PICKLE_ROOT, 'samples/df_train100_first10.pkl'),
        info_loc= os.path.join(PICKLE_ROOT, 'video_info.pkl')
        )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

class Trainer(object):
    def __init__(self, verb_loc, noun_loc, optimizer, loss_fn, df_train) -> None:
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.df_train = df_train
        
        self.train_loader = get_dataloader()
        self.val_loader = get_dataloader(train=False)
        self.device = get_device()
        
        self.embedding_model = WordEmbeddings()
        self.attention_model = AttentionModel(
                        self.verb_map,
                        self.noun_map
                    )
        
        self.verb_map = self.get_word_map(verb_loc)
        self.noun_map = self.get_word_map(noun_loc)
        self.verb_embeddings = self.get_embeddings('verb')
        self.noun_embeddings = self.get_embeddings('noun')
        self.verb_one_hot = F.one_hot(torch.arange(0, NUM_VERBS))
        self.noun_one_hot = F.one_hot(torch.arange(0, NUM_NOUNS))

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

    def get_embeddings(self, mode):
        if mode == 'verb':
            text = self.verb_map['key'].values.tolist()
        elif mode == 'noun':
            text = self.noun_map['key'].values.tolist()
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')   
        
        embeddings = self.embedding_model(text)
        return embeddings

    def get_word_map(self, file_loc):
        try:
            with open(file_loc, 'rb') as handle:
                df = pickle.load(handle)
        except:
            raise FileNotFoundError(f'Invalid pickle location: {file_loc}')
        
        return df[['id', 'key']]
    
    def _train(self, train=True):
        if train:
            self.attention_model.train()
            loader = self.train_loader
        else:
            self.attention_model.eval()
            loader = self.val_loader

        train_loss_meter = ActionMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # loop over each minibatch
        for (video_id, frame_id, feats) in tqdm(loader):
            feats = feats.to(self.device)
            n = feats.shape[0]
            verb_class = self.df_train[self.df_train['video_id'] == video_id & self.df_train['frame_id'] == frame_id]['verb_class']
            noun_class = self.df_train[self.df_train['video_id'] == video_id & self.df_train['frame_id'] == frame_id]['noun_class']

            #* assert feats.size() == [b, WORD_EMBEDDING_SIZE]
            predictions_verb, predictions_noun = self.attention_model(feats, verb_class, noun_class)

            batch_acc_noun = self.compute_accuracy(predictions_noun, noun_class)
            batch_acc_verb = self.compute_accuracy(predictions_verb, verb_class)
            train_acc_meter.update(batch_acc_noun, batch_acc_verb, n=n)

            # Pytorch multi-loss reference: https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
            batch_loss = self.compute_loss(predictions_verb, self.verb_one_hot[verb_class]) + self.compute_loss(predictions_noun, self.noun_one_hot[noun_class])
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            if train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        return train_loss_meter.avg, train_acc_meter.avg_verb, train_acc_meter.avg_noun

    def compute_accuracy(self, preds, labels):
        preds = torch.argmax(preds, dim=1)
        correct = (preds == labels).float().sum().item()
        batch_accuracy = correct / len(preds)

        return batch_accuracy

    def compute_loss(self, preds, labels, is_normalize=False):
        loss = self.loss_fn(preds, labels)
        if is_normalize:
            loss = loss / len(preds)

        return loss

    # baseline paper used num_epochs = 3e6
    def training_loop(self, num_epochs=500, train=True):
        """Run the main training loop for the model.
        May need to run separate loops for train/test.

        Args:
            num_epochs (int, optional): _description_. Defaults to 500.
            test (bool, optional): _description_. Defaults to False.
        """
        for epoch in tqdm(range(num_epochs)):
            train_loss, verb_acc, noun_acc = self._train()

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append((verb_acc, noun_acc))

            # val_loss, val_verb_acc, val_noun_acc = self._train(train=False)
            # self.validation_loss_history.append(val_loss)
            # self.validation_accuracy_history.append((val_verb_acc, val_noun_acc))

            print(
                f"Epoch:{epoch + 1}"
                + f" Train Loss:{train_loss:.4f}"
                # + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy (verb/noun): {verb_acc:.4f}/{noun_acc:.4f}"
                # + f" Validation Accuracy (verb/noun): {val_verb_acc:.4f}/{val_noun_acc:.4f}"
            )


if __name__ == '__main__':
    loader = get_dataloader()
    print(f'Obtained dataloader: length = {len(loader)}')
    # for (v, f, feats) in loader:
    #     print(feats.shape)
    
    with open(os.path.join(PICKLE_ROOT, 'samples/df_train100_first10.pkl'), 'rb') as handle:
        df = pickle.load(handle)
    optimizer = torch.optim.Adam(lr=1e-5, weight_decay=1e-5)
    trainer = Trainer(VERB_CLASSES, NOUN_CLASSES, optimizer, nn.CrossEntropyLoss(), df)
    trainer.training_loop(num_epochs=1)
