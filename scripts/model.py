import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_feature_length):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 100, 3),
            nn.ReLU()
        )
        self.attention = None
