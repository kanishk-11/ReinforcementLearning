

import logging
from typing import Dict
from numpy import dtype
import numpy
import torch

from main.normalizers import PytorchRunningMeanStd, RunningMeanStd


class RNDLifeLongBonusModule:
    def __init__(
        self,
        target_network : torch.nn.Module,
        predictor_network : torch.nn.Module,
        device : torch.device,
        discount : float
    ):
        self._target_network = target_network
        self._predictor_network = predictor_network
        self._device = device
        self._discount = discount
        self._intrinsic_reward_normalizer = RunningMeanStd((1,))
        self._RND_observation_normalizer = PytorchRunningMeanStd((1,84,84),device=self._device)
    
    @torch.no_grad()
    def _normalize_rnd_observation(self, rnd_observation):
        rnd_observation = rnd_observation.to(device = self._device,dtype= torch.float32)
        normed_observation = self._RND_observation_normalizer.normalize(rnd_observation)
        normed_observation = normed_observation.clamp(-5,5)
        self._RND_observation_normalizer.update_single(rnd_observation)
        return normed_observation
    
    def normalize_intrinsic_reward(self,intrinsic_reward):
        # logging.info(type(intrinsic_reward))
        self._intrinsic_reward_normalizer.update_single(intrinsic_reward)
        normed_intrinsic_reward = intrinsic_reward/numpy.sqrt(self._intrinsic_reward_normalizer.var + 1e-8)
        return normed_intrinsic_reward.item()
    @torch.no_grad()
    def get_bonus(self,state_t:torch.Tensor):
        normed_state_t = self._normalize_rnd_observation(state_t)
        prediction = self._predictor_network(normed_state_t)
        target = self._target_network(normed_state_t)
        # logging.info(prediction,target)
        intrinsic_reward = torch.square(prediction-target).mean(dim=-1).detach().cpu().numpy()
        normed_intrinsic_reward_t = self.normalize_intrinsic_reward(intrinsic_reward)
        return normed_intrinsic_reward_t
    def update_predictor_network(self,state_dict:Dict):
        self._predictor_network.load_state_dict(state_dict)