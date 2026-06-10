import torch
from torch import nn

class EvalNet(nn.Module):
    def __init__(self, input_size: int = 769, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        return out
