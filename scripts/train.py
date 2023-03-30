import os
from frame_loader import FrameLoader
from constants import BATCH_SIZE, PICKLE_ROOT

import pandas as pd
import pickle

from torch.utils.data import DataLoader
from models import AttentionModel, WordEmbeddings

from utils import AverageMeter, get_device


def get_dataloader():
    dataset = FrameLoader(
        loc = os.path.join(PICKLE_ROOT, 'samples/df_train100.pkl'),
        info_loc= os.path.join(PICKLE_ROOT, 'video_info.pkl')
        )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    loader = get_dataloader()
    for (v, f, feats) in loader:
        print(feats.shape)


class Trainer:
    def __init__(self, verb_loc, noun_loc, optimizer, loss_fn, train=True, num_epochs = 500) -> None:
        self.loader = get_dataloader()
        self.verb_map = self.get_word_map(verb_loc)
        self.noun_map = self.get_word_map(noun_loc)
        self.opt = optimizer
        self.loss_fn = loss_fn
        self.train = train
        self.num_epochs = num_epochs
        self.device = get_device()
        
        self.embedding_model = WordEmbeddings()
        self.model = AttentionModel(
                        len(self.verb_map), 
                        len(self.noun_map),
                        self.verb_map,
                        self.noun_map
                    )
        
        self.verb_embeddings = self.get_embeddings('verb')
        self.noun_embeddings = self.get_embeddings('noun')

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
    
    def _train(self, loader, df_train, optimizer, loss_fn, test=False, lr=0.01):
        train_loss_meter = AverageMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # loop over each minibatch
        for (video_id, frame_id, feats) in loader:
            feats = feats.to(self.device)
            verb = df_train[df_train['video_id'] == video_id & df_train['frame_id'] == frame_id]['verb']
            noun = df_train[df_train['video_id'] == video_id & df_train['frame_id'] == frame_id]['noun']

            predictions_verb, predictions_noun = self.model(feats, verb, noun)

            batch_acc = compute_accuracy(logits, y)
            train_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        return train_loss_meter.avg, train_acc_meter.avg

    #! copy from CV proj 4
    def compute_accuracy(self):
        pass

    def compute_loss(self):
        pass
        
        

    #! Need to rewrite - see example for CV Proj 4
    def training_loop(loader, optimizer, loss_fn, num_epochs=500, test=False, lr=0.01):
        """Run the main training loop for the model.
        May need to run separate loops for train/test.

        Args:
            loader (_type_): _description_
            optimizer (_type_): _description_
            loss_fn (_type_): _description_
            num_epochs (int, optional): _description_. Defaults to 500.
            test (bool, optional): _description_. Defaults to False.
            lr (float, optional): _description_. Defaults to 0.01.
        """
        pass

    '''
    f = [b, L, 384]
    v = [97, 384]
    res = [b, L, 97]
    '''

