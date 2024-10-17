import logging
import numpy as np
import torch

from main.networks.common import BackboneConv, get_dimesions_after_conv, init_weights_of_net
from models.common import disable_autograd


class RNDConv(torch.nn.Module):
    def __init__(
        self,
        state_dim:int,
        is_target:bool,
        latent_dim:int = 256
    ):
        super().__init__()
        self.state_dim = state_dim
        self.is_target = is_target
        self.latent_dim = latent_dim
        c,h,w = state_dim
        h,w = get_dimesions_after_conv((h,w),8,4)
        h,w = get_dimesions_after_conv((h,w),4,2)
        h,w = get_dimesions_after_conv((h,w),3,1)
        out_size = 64*h*w
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten()
        )
        if is_target:
            self.head = torch.nn.Linear(out_size,latent_dim)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(out_size,512),
                torch.nn.ReLU(),
                torch.nn.Linear(512,512),
                torch.nn.ReLU(),
                torch.nn.Linear(512,latent_dim)
            )
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()

        if is_target:
            disable_autograd(self)
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x

class NGUEmbedding(torch.nn.Module):
    def __init__(self,state_dim:int,action_dim:int,embedding_size:int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_size = embedding_size
        self.net = BackboneConv(state_dim)
        self.head = torch.nn.Linear(self.net.out_features,self.embedding_size)
        self.inverse_head = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size*2,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,action_dim),
            torch.nn.ReLU()
        )
        init_weights_of_net(self)
    def forward(self,x):
        x = x.float()/255.0
        x = self.net(x)
        x = self.head(x)
        return x
    def predict_action_logits(self,x):
        return self.inverse_head(x)
        