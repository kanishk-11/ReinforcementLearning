
from typing import NamedTuple
import torch
class ActorNetworkOutput(NamedTuple):
    pi_logits: torch.Tensor  # [batch_size, action_dim]
class CriticNetworkOutput(NamedTuple):
    value: torch.Tensor

class ActorMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )
    def forward(self,input):
        pi_logits = self.net(input)
        return ActorNetworkOutput(pi_logits = pi_logits)
class CriticMLP(torch.nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
            torch.nn.ReLU()
        )
    def forward(self, input):
        value= self.body(input)
        return CriticNetworkOutput(value=value)
        