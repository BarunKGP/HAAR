import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from constants import WORD_EMBEDDING_SIZE, MULTIMODAL_FEATURE_SIZE
from utils import get_device, vector_gather

# Neural Network configs
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())

# WORD EMBEDDINGS
class WordEmbeddings(nn.Module):
    def __init__(self, model=embedding_model) -> None:
        super().__init__()
        self.model = model
        self.device = get_device()

    def forward(self, text):
        embeddings = self.model.encode(text)
        return torch.from_numpy(embeddings)

    
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
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(MULTIMODAL_FEATURE_SIZE, WORD_EMBEDDING_SIZE)
        )
        self.linear_verb = nn.Linear(WORD_EMBEDDING_SIZE, self.C_verb, bias=True)
        self.linear_noun = nn.Linear(WORD_EMBEDDING_SIZE, self.C_noun, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def _predictions(self, frame_features, key, mode) -> torch.Tensor:
        """Takes the frame_features and returns the predictions
        for action (verb/noun)
        ------ Shape logic ------
        f = [b, D, 100]
        w1 = [b, C, D]
        A = w1@f = [b, C, 100]
        Ai = A.T[:, verb, :] = [b, 1, 100]
        F_i = Ai @ f.permute(0, 2, 1) = [b, 1, D]
        P = W2(F_i) = [b, 1, C]
        res = softmax(P) = [b, 1, C]
        ----------------------------

        Args:
            frame_features (torch.Tensor): the multimodal features
            key (torch.Tensor): tensor of verb/noun class indices 
                used to collect class-aware attention
            mode (str): whether to predict verb or noun

        Raises:
            Exception: Invalid mode. It has to be 'verb' or 'noun'

        Returns:
            torch.Tensor: prediction probabilities for each verb/noun
                class
        """
        if mode == 'verb':
            embeddings = self.verb_embeddings.to(self.device)
            linear_layer = self.linear_verb
        elif mode == 'noun':
            embeddings = self.noun_embeddings.to(self.device)
            linear_layer = self.linear_noun
        else:
            raise Exception('Invalid mode: choose either "noun" or "verb"')        
        
        if embeddings.ndim == 1:
            embeddings = embeddings[:, None] # Convert to shape [b, K]
        
        A = torch.matmul(embeddings, frame_features)
        A = torch.sigmoid(A) # shape: [b, C, 100]
        A = vector_gather(A, key)
        y = torch.einsum('ijk, ik -> ij', frame_features, A)
        y = torch.div(y, torch.sum(A, dim=-1).reshape((-1, 1))) 
        y = linear_layer(y)
        y = self.softmax(y)

        return y
    
    def forward(self, x: torch.Tensor, verb_class, noun_class):
        x = x[:, None, :].to(torch.float32)
        x = self.layer1(x).permute((0, 2, 1))
        verb_predictions = self._predictions(x, verb_class, 'verb')
        noun_predictions = self._predictions(x, noun_class, 'noun')

        return verb_predictions, noun_predictions