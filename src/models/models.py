import torch
import torch.nn as nn
from constants import (
    MULTIMODAL_FEATURE_SIZE,
    SENTENCE_TRANSFORMER_MODEL,
    WORD_EMBEDDING_SIZE,
)
from sentence_transformers import SentenceTransformer
from utils import get_device, vector_gather

# Neural Network configs
embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=get_device())


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
    def __init__(self, embeddings, word_map):
        super().__init__()

        self.embeddings = embeddings
        self.word_map = word_map
        self.cardinality = len(self.word_map)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(MULTIMODAL_FEATURE_SIZE, WORD_EMBEDDING_SIZE),
        )
        self.linear_layer = nn.Linear(WORD_EMBEDDING_SIZE, self.cardinality, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def _predictions(self, frame_features, key):
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

        Raises:
            Exception: Invalid mode. It has to be 'verb' or 'noun'

        Returns:
            torch.Tensor: prediction probabilities for each verb/noun
                class
        """
        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings[:, None]  # Convert to shape [b, K]
        A = torch.sigmoid(
            torch.matmul(self.embeddings.to(self.device), frame_features)
        )  # shape: [b, C, 100]

        #! class-aware attention should only be done in training, figure out different flow for testing
        #! This is true for training only
        y = None
        if key is not None:
            A = vector_gather(A, key)
            y = torch.einsum("ijk, ik -> ij", frame_features, A)
            y = torch.div(y, torch.sum(A, dim=-1).reshape((-1, 1)))
            y = self.linear_layer(y)
            y = self.softmax(y)
        del A
        return y

    def _evaluate(self):
        pass

    def forward(self, x: torch.Tensor, label: int):
        x = x[:, None, :].to(torch.float32)
        x = self.layer1(x).permute((0, 2, 1))
        return self._predictions(x, label)
