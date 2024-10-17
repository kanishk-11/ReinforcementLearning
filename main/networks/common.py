
import math
from typing import Tuple

import torch

def get_dimesions_after_conv(h_w:Tuple,kernel_size:int=1,stride:int=1,pad:int=0,dilation:int=1)->Tuple[int,int]:
    h_return = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    w_return = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)           
    return (h_return,w_return)

def init_weights_of_net(net:torch.nn.Module):
    assert isinstance(net, torch.nn.Module)

    for module in net.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')


class BackboneConv(torch.nn.Module):
    def __init__(self, state_dim:Tuple) -> None:
        super().__init__()
        out_channels = 64
        c,h,w = state_dim
        h,w = get_dimesions_after_conv((h,w),8,4)
        h,w = get_dimesions_after_conv((h,w),4,2)
        h,w = get_dimesions_after_conv((h,w),3,1)
        
        self.out_features = out_channels*h*w
        
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c , out_channels=128,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=64,kernel_size=4,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=out_channels,kernel_size=3,stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
    def forward(self,input:torch.Tensor):
        return self.net(input)