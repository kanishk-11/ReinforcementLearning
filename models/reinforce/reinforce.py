
import collections
from os import times
import time
from typing import Tuple
import numpy as np
from sympy import sequence
import torch
from main.ReplayUtils.common import stack_transitions
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
from main.ReplayUtils.transition import Transition
from main.abstractClasses import action
from main.abstractClasses.action import Action
from main.abstractClasses.agent import Agent
from main.abstractClasses.timestep import TimeStepPair
from main.distributions import categorical_distribution
from main.loss import losses
from main.loss.losses import policy_gradient_loss


class ReinforceWithBaseline(Agent):
    def __init__(
        self,
        policy_network:torch.nn.Module,
        policy_optimizer:torch.optim.Optimizer,
        discount:float,
        value_network:torch.nn.Module,
        baseline_optimizer:torch.optim.Optimizer,
        transition_accumulator:TransitionAccumulator,
        normalize_returns:bool,
        clip_gradients:bool,
        max_gradient_norm:float,
        device: torch.device
    ):
        if not 0.0<=discount<=1.0:
            raise ValueError(f'Expect discount to be in range [0, 1], got {discount}')
        self.agent_name = "ReinforceWithBaseline"
        self._device = device
        self._policy_network = policy_network.to(device)
        self._policy_optimizer = policy_optimizer
        self._discount = discount
        self._value_network = value_network.to(device)
        self._baseline_optimizer = baseline_optimizer
        self._transition_accumulator = transition_accumulator
        # Needed as it is a monte carlo exstimator
        self._trajectory = collections.deque(maxlen=108000)
        
        self._normalise_returns = normalize_returns
        self._clip_gradients = clip_gradients
        self._max_gradient_norm = max_gradient_norm
        
        self._step_t = -1
        self._update_t = 0
        self._policy_loss_t = np.nan
        self._value_loss_t = np.nan
    def step(self,timestep:TimeStepPair) -> Action:
        self._step_t +=1
        action_t = self.act(timestep)
        for transition in self._transition_accumulator.step(timestep_t=timestep,action_t=action_t):
            self._trajectory.append(transition)
        if timestep.done:
            self._learn()
        return action_t
    def reset(self):
        self._transition_accumulator.reset()
        self._trajectory.clear()
    def act(self, timestep:TimeStepPair) -> Action:
        action_t = self._choose_action(timestep)
        return action_t
    @torch.no_grad()
    def _choose_action(self, timestep:TimeStepPair) -> Action:
        state_t = torch.from_numpy(timestep.observation[None,...]).to(device=self._device,dtype=torch.float32)
        logits = self._policy_network(state_t).pi_logits
        action_t = categorical_distribution(logits).sample()
        return action_t.cpu().item()
    def _learn(self):
        transitions = stack_transitions(list(self._trajectory),Transition(None,None,None,None,None))
        self._update(transitions)
    def _update(self, transitions:Transition):
        self._baseline_optimizer.zero_grad()
        self._policy_optimizer.zero_grad()
        polilcy_loss, value_loss = self._calculate_loss(transitions)
        value_loss.backward()
        if self._clip_gradients:
            torch.nn.utils.clip_grad_norm_(self._value_network.parameters(), self._max_gradient_norm, error_if_nonfinite=True)
        self._baseline_optimizer.step()
        
        polilcy_loss.backward()
        if self._clip_gradients:
            torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), self._max_gradient_norm, error_if_nonfinite=True)
        self._policy_optimizer.step()
        self._update_t +=1
    def _calculate_loss(self, transitions: Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        state_t_minus_1 = torch.from_numpy(transitions.state_t_minus_1).to(device=self._device,dtype=torch.float32)
        action_t_minus_1 = torch.from_numpy(transitions.action_t_minus_1).to(device=self._device,dtype=torch.int64)
        reward_t = torch.from_numpy(transitions.reward_t).to(device=self._device,dtype=torch.float32)
        done = torch.from_numpy(transitions.done).to(device=self._device,dtype=torch.bool)
        
        discount = (~done).float() * self._discount
        sequence_length = len(reward_t)
        returns = torch.empty(sequence_length,device=self._device)
        g = 0.0
        for t in reversed(range(sequence_length)):
            g = reward_t[t] + discount[t] * g
            returns[t] = g
        logits_t_minus_1 = self._policy_network(state_t_minus_1).pi_logits
        value_t_minus_1 = self._value_network(state_t_minus_1).value.squeeze(1)
        
        delta = returns - value_t_minus_1
        
        policy_loss = policy_gradient_loss(logits_t_minus_1,action_t_minus_1,delta).loss
        value_loss = losses.value_loss(returns,value_t_minus_1).loss
        
        value_loss = torch.mean(value_loss,dim=0)
        policy_loss = torch.mean(policy_loss,dim=0)
        self._value_loss_t = value_loss.detach().cpu().item()
        self._policy_loss_t = policy_loss.detach().cpu().item()
        return policy_loss, value_loss
    @property
    def statistics(self):
        return {
            'value_loss':self._value_loss_t,
            'policy_loss':self._policy_loss_t,
            'updates':self._update_t
        }