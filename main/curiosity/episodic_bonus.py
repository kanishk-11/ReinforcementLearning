import dis
import logging
from turtle import distance
from typing import Dict, NamedTuple
import torch

from main.normalizers import PytorchRunningMeanStd
class KNNQueryResult(NamedTuple):
    neighbors:torch.Tensor
    neighbor_indices:torch.Tensor
    neighbor_distances:torch.Tensor
    
def knn_query(embedding,memory,num_neighbors):
    assert memory.shape[0] >= num_neighbors
    # logging.info(embedding.shape)
    # logging.info(memory.shape)
    distances = torch.cdist(embedding.unsqueeze(0),memory).squeeze(0).pow(2)
    distances,indices = distances.topk(num_neighbors,largest=False)
    neighbors = torch.stack([memory[ind] for ind in indices],dim=0)
    return KNNQueryResult(neighbors = neighbors,neighbor_indices=indices,neighbor_distances=distances)
class EpisodicBonusModule:
    def __init__(
        self,
        embedding_network : torch.nn.Module,
        device : torch.device,
        size : int,
        num_neighbors : int,
        kernel_epsilon : float,
        cluster_distance : float,
        maximum_similarity : float,
        c_constant : float =0.001
    ) -> None:
        self._embedding_network = embedding_network.to(device=device)
        self._device = device
        self._memory = torch.zeros(size, self._embedding_network.embedding_size,device=self._device)
        self._mask = torch.zeros(size, dtype=torch.bool,device=self._device)
        self._size = size
        self._counter = 0
        self._cluster_distance_normalizer = PytorchRunningMeanStd(shape=(1,),device=self._device)
        self._num_neighbors = num_neighbors
        self._kernel_epsilon = kernel_epsilon
        self._cluster_distance = cluster_distance
        self._max_similarity = maximum_similarity
        self._c_constant = c_constant
    def _add_to_memory(self,embedding:torch.Tensor):
        index = self._counter %  self._size
        self._memory[index] = embedding
        self._mask[index] = True
        self._counter += 1
    @torch.no_grad()
    def get_bonus(self,state_t:torch.Tensor)->torch.Tensor:
        embedding = self._embedding_network(state_t).squeeze(0)
        previous_mask = self._mask
        self._add_to_memory(embedding)
        if self._counter <= self._num_neighbors:
            return 0.0
        knn_query_result = knn_query(embedding,self._memory[previous_mask],self._num_neighbors)
        nn_distance_squared = knn_query_result.neighbor_distances
        self._cluster_distance_normalizer.update_single(nn_distance_squared)
        distance_rate = nn_distance_squared/(self._cluster_distance_normalizer.mean + 1e-8)
        distance_rate = torch.max((distance_rate - self._cluster_distance),torch.tensor(0.0))
        kernel_output = self._kernel_epsilon/(distance_rate + self._kernel_epsilon)
        similarity = torch.sqrt(torch.sum(kernel_output)) + self._c_constant
        if torch.isnan(similarity):
            return 0.0
        if similarity>self._max_similarity:
            return 0.0
        return (1/similarity).cpu().item()
    def reset(self,reset_memory=True):
        self._mask = torch.zeros(self._size,dtype=torch.bool,device=self._device)
        self._counter = 0
        if reset_memory:
            self._memory=torch.zeros(self._size, self._embedding_network.embedding_size,device=self._device)
    def update_embedding_network(self,state_dict:Dict):
        self._embedding_network.load_state_dict(state_dict)