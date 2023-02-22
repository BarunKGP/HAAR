import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1024, 3),
            nn.ReLU(),
            nn.Conv1d(1024, 100, 3),
            nn.ReLU()
        )
        self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_output = None
        frame_features = self.conv1(x)

        # TODO: Complete attention module


        return  model_output

        