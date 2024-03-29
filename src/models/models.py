import torch
import torch.nn as nn
from constants import (
    MULTIMODAL_FEATURE_SIZE,
    SENTENCE_TRANSFORMER_MODEL,
    WORD_EMBEDDING_SIZE,
    D_MODEL_ROOT
)
from sentence_transformers import SentenceTransformer
from utils import get_device, vector_gather

# WORD EMBEDDINGS
class WordEmbeddings(nn.Module):
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=self.device)

    def forward(self, text):
        embeddings = self.model.encode(text)
        return torch.from_numpy(embeddings)


# ATTENTION MODEL
class AttentionModel(nn.Module):
    def __init__(self, word_map, d_model_root=D_MODEL**0.5):
        super().__init__()
        self.cardinality = len(word_map)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(MULTIMODAL_FEATURE_SIZE, WORD_EMBEDDING_SIZE),
        )
        self.linear_layer = nn.Linear(WORD_EMBEDDING_SIZE, self.cardinality, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        #! Need to define MODEL_OUT, h
        d_v = MODEL_OUT/h
        self.linear_Q = nn.Linear(D_MODEL, d_v)
        self.linear_K = nn.Linear(D_MODEL, d_v)
        self.linear_V = nn.Linear(D_MODEL, d_v)
        self.linear_out =  = nn.Linear(h*d_v, MODEL_OUT)

    def _predictions(self, frame_features, key, embeddings):
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
        if embeddings.ndim == 1:
            embeddings = embeddings[:, None]  # Convert to shape [b, K]
        A = torch.sigmoid(
            torch.matmul(embeddings, frame_features)
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

    # def _sdpa(self, q, k, v):
    #     """Scaled dot-product attention""" 
    #     return torch.matmul(
    #         self.softmax(
    #             torch.matmul(q, k.T)/D_MODEL_ROOT 
    #         ),
    #         v
    #     )

    # def multihead_attention(self, q, k, v, h):
    #     attention_heads = []
    #     for head in range(h):
    #         attention_heads.append(self._sdpa(q, k, v))


    def forward(self, x: torch.Tensor, label: int, embeddings):
        x = x[:, None, :].to(torch.float32)
        x = self.layer1(x).permute((0, 2, 1))
        return self._predictions(x, label, embeddings)
