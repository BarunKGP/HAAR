import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from constants import WORD_EMBEDDING_SIZE
from utils import get_device

# Neural Network configs
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())

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
    def __init__(self, verb_embeddings, noun_embeddings, verb_map, noun_map):
        super().__init__()

        self.verb_embeddings = verb_embeddings
        self.noun_embeddings = noun_embeddings
        self.verb_map = verb_map
        self.noun_map = noun_map
        
        self.C_verb = len(self.verb_map)
        self.C_noun = len(self.noun_map)
        self.device = get_device()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, device=self.device),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4094, WORD_EMBEDDING_SIZE)
        )
        self.linear_verb = nn.Linear(WORD_EMBEDDING_SIZE, self.C_verb, bias=True)
        self.linear_noun = nn.Linear(WORD_EMBEDDING_SIZE, self.C_noun, bias=True)
        self.softmax = nn.Softmax()

    def _predictions(self, frame_features, key, mode) -> torch.Tensor:
        """_summary_

        ------ Shape logic ------
        f = [b, 100, D]
        w1 = [b, C, D]
        A = f @ w1 = [b, 100, C]
        Ai = A.T[verb] = [100, b]
        F_i = Ai.permute(0, 2, 1) @ f = [b, D, b]
        P = W2(F_i).T = [b, C]
        res = softmax(P) = [b, C]
        --------------------------
        f = [b, D, 100]
        w1 = [b, C, D]
        A = w1@f = [b, C, 100]
        Ai = A.T[:, verb, :] = [b, 1, 100]
        F_i = Ai @ f.permute(0, 2, 1) = [b, 1, D]
        P = W2(F_i) = [b, 1, C]
        res = softmax(P) = [b, 1, C]
        ----------------------------

        Args:
            frame_features (_type_): _description_
            key (_type_): _description_
            mode (_type_): _description_

        Raises:
            Exception: _description_

        Returns:
            torch.Tensor: _description_
        """
        if mode == 'verb':
            embeddings = self.verb_embeddings.to(self.device)
            linear_layer = self.linear_verb
        elif mode == 'noun':
            embeddings = self.noun_embeddings.to(self.device)
            linear_layer = self.linear_noun
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')        
        
        # attention = torch.sigmoid(torch.sum(embeddings * frame_features.T, dim=-1)) # hacky way to do rowwise dot product. Link: https://stackoverflow.com/questions/61875963/pytorch-row-wise-dot-product
        attention = torch.sigmoid(torch.matmul(embeddings, frame_features)) # shape: [b, C, 100]
        aware = attention[:, key:, ]
        weighted_features = torch.matmul(aware, frame_features.permute(0, 2, 1))/torch.sum(aware, dim=-1) 
        predictions = linear_layer(weighted_features)
        predictions = self.softmax(predictions)

        return predictions
    
    def forward(self, x: torch.Tensor, verb_class, noun_class):
        x = x[:, None, :]
        print(f'x.shape = {x.size()}')
        frame_features = self.layer1(x)
        verb_predictions = self._predictions(frame_features, verb_class, 'verb').detach().cpu()
        noun_predictions = self._predictions(frame_features, noun_class, 'noun').detach().cpu()

        return verb_predictions, noun_predictions