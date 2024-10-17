

from asyncio.log import logger
from typing import NamedTuple, Optional

import torch


class LossAndExtra(NamedTuple):
    loss: torch.Tensor
    extraInformation:Optional[NamedTuple]

class QLearningExtra(NamedTuple):
    target:Optional[torch.Tensor]
    td_error:Optional[torch.Tensor]
class DDQLearningExtra(NamedTuple):
    target:Optional[torch.Tensor]
    td_error:Optional[torch.Tensor]
    best_action:Optional[torch.Tensor]
def get_batched_index(values: torch.Tensor, indices: torch.Tensor, dim: int = -1, keepdims: bool = False) -> torch.Tensor:
    indices = indices.long()   
    one_hot_indices = torch.nn.functional.one_hot(indices,values.shape[dim]).to(dtype = values.dtype)
    if len(values.shape) == 3 and len(one_hot_indices.shape) == 2:
        one_hot_indices = one_hot_indices.unsqueeze(1)
    return torch.sum(values*one_hot_indices,dim=dim,keepdims=keepdims)

    