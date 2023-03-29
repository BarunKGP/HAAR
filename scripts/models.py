import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from scripts.constants import WORD_EMBEDDING_SIZE
from scripts.utils import AverageMeter, get_device

# Neural Network configs
device = get_device()
conv1 = nn.Sequential(
            nn.Conv1d(1, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, WORD_EMBEDDING_SIZE, 3), # alt. strategy: keep 100 neurons and pad to match shape of word embeddings
            nn.ReLU()
        )

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# WORD EMBEDDINGS
class WordEmbeddings(nn.Module):
    def __init__(self, model=embedding_model) -> None:
        super().__init__()
        self.model = model
        self.device = get_device()

    def forward(self, text):
        embedding = self.model.encode(text)
        return embedding

    
# ATTENTION MODEL
class AttentionModel(nn.Module):
    def __init__(self, verb_embeddings, noun_embeddings, C_verb, C_noun, verb_map, noun_map):
        super().__init__()

        self.verb_embeddings = verb_embeddings
        self.noun_embeddings = noun_embeddings
        self.C_verb = C_verb
        self.C_noun = C_noun
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.device = get_device()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, WORD_EMBEDDING_SIZE, 3),
            nn.ReLU()
        )
        self.linear_verb = nn.Linear(WORD_EMBEDDING_SIZE, self.C_verb, bias=True)
        self.linear_noun = nn.Linear(WORD_EMBEDDING_SIZE, self.C_noun, bias=True)
        self.softmax = nn.Softmax()

    def _predictions(self, frame_features, mode) -> torch.Tensor:
        if mode == 'verb':
            embeddings = self.verb_embeddings
            word_map = self.verb_map
            linear_layer = self.linear_verb
        elif mode == 'noun':
            embeddings = self.noun_embeddings
            word_map = self.noun_map
            linear_layer = self.linear_noun
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')        
        
        attention = torch.sigmoid(torch.sum(embeddings * frame_features.T, dim=-1)) # hacky way to do rowwise dot product. Link: https://stackoverflow.com/questions/61875963/pytorch-row-wise-dot-product
        weighted_features = (attention * frame_features)/torch.sum(attention, dim=-1)
        predictions = linear_layer(weighted_features).T
        predictions = self.softmax(predictions).cpu()

        return word_map[word_map.id == torch.argmax(predictions)]
    
    def forward(self, x: torch.Tensor, verb, noun):
        x = x.to(self.device)
        frame_features = self.conv1(x)
        verb_predictions = self._predictions(frame_features, 'verb')
        noun_predictions = self._predictions(frame_features, 'noun')

        return verb_predictions, noun_predictions