import copy
import logging
from typing import Callable, overload
from venv import logger

import numpy as np
import torch
from main.ReplayUtils.replayBuffers import UniformReplay
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
from main.ReplayUtils.transition import Transition
from main.abstractClasses.agent import Agent 
from main.abstractClasses.timestep import TimeStepPair
from main.abstractClasses.action import Action
from torch import nn 
import main.loss.losses as loss_lib
class DQN(Agent):
    """DQN Agent Class"""
    def __init__(
            self,
            network : nn.Module,
            optimizer : torch.optim.Optimizer,
            random_state : np.random.RandomState,
            replay_buffer : UniformReplay,
            transition_accumulator : TransitionAccumulator,
            exploration_rate_lambda: Callable[[int],float],
            learning_interval: int,
            target_network_update_interval: int,
            min_replay_size:int,
            batch_size:int,
            action_space_dimension:int,
            discount:float,
            clip_gradient:bool,
            max_gradient_norm: float,
            device:torch.device
    ):
        super().__init__()

        if not 1 <= learning_interval:
            raise ValueError(f'learning_interval should be a positive integer, got {learning_interval}')
        if not 1 <= target_network_update_interval:
            raise ValueError(f'target_network_update_interval should be a positive integer, got {target_network_update_interval}')
        if not min_replay_size > 0:
            raise ValueError(f'min_replay_size should be a positive integer, got {min_replay_size}')
        if not batch_size<= min_replay_size <= replay_buffer.size:
            raise ValueError(f' batch_size<= min_replay_size <= replay_buffer.size condition has to be satisfied')
        if not 0 < action_space_dimension:
            raise ValueError(f'action_space_dimension need to be a positive integer')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'dicount needs to be between 0 and 1')
        self._setup_dqn_meta(device,random_state)
        self._setup_dqn_networks_and_optimizer(network,optimizer)
        self._setup_replay(transition_accumulator,batch_size,replay_buffer)
        self._setup_learning_parameter(discount,exploration_rate_lambda,min_replay_size,learning_interval,target_network_update_interval,clip_gradient,max_gradient_norm)
        # Setup for DQN itself
        self._action_space_dim = action_space_dimension
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def _setup_dqn_meta(self,device:torch.device,random_state:np.random.RandomState):
        self.agent_name = "DQN"
        self._device = device
        self._random_state = random_state
    def _setup_dqn_networks_and_optimizer(self,network:nn.Module,optimizer:torch.optim.Optimizer):
        self._main_network = network.to(self._device)
        self._target_network = copy.deepcopy(self._main_network).to(self._device)
        self._optimizer = optimizer
        for p in self._target_network.parameters():
            p.requires_grad = False
    def _setup_replay(self,transition_accumulator:TransitionAccumulator,batch_size:int,replay_buffer:UniformReplay):
        self._transaction_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._replay_buffer = replay_buffer
    def _setup_learning_parameter(self,discount:float,exploration_rate_lambda:Callable[[int],float],min_replay_size:int,learning_interval:int,target_network_update_interval:int,clip_gradient:bool,max_gradient_norm:float):
        self._exploration_rate_lambda = exploration_rate_lambda
        self._clip_gradient = clip_gradient
        self._discount = discount
        self._min_replay_size = min_replay_size
        self._learning_interval = learning_interval
        self._target_network_update_interval = target_network_update_interval
        self._max_gradient_norm = max_gradient_norm
    def step(self,timestep:TimeStepPair) -> Action:
        self._step_t +=1

        action_t = self.act(timestep)

        for transition in self._transaction_accumulator.step(timestep,action_t):
            if transition is not None:
                self._replay_buffer.add(transition)
        
        if self._replay_buffer.transitions_added < self._min_replay_size:
            return action_t
        
        if self._step_t % self._learning_interval == 0:
            self.learn()
        
        return action_t
    
    def act(self,timestep:TimeStepPair):
        """get action by running it through the network"""
        return self._choose_action(timestep,self.exploration_rate)
    
    @torch.no_grad()
    def _choose_action(self,timestep:TimeStepPair,exploration_rate:float) -> Action:
        if self._random_state.rand() <= exploration_rate:
            return self._random_state.randint(0,self._action_space_dim)
        else:
            state = torch.from_numpy(timestep.observation[None,...]).to(device=self._device,dtype=torch.float32)
            Q_t = self._main_network(state).q_values
            action_t = torch.argmax(Q_t,dim=-1)
            return Action(action_t.cpu().item())
    
    def learn(self):
        # logger.info(f'replay buffer size:{self._replay_buffer.size} , min_replay_size : {self._min_replay_size}, transitions_filled:{self._replay_buffer.transitions_added}')
        transitions = self._replay_buffer.sample(self._batch_size)
        self._update(transitions)
        if self._update_t > 1 and self._update_t % self._target_network_update_interval==0:
            self._update_target_network()
    
    def _update(self,transitions):
        #Clear optimizer gradients
        self._optimizer.zero_grad()
        loss = self._calculate_loss(transitions)
        # logging.info("got loss")
        loss.backward()
        
        if self._clip_gradient:
            torch.nn.utils.clip_grad.clip_grad_norm_(self._main_network.parameters(),self._max_gradient_norm,error_if_nonfinite=True)
        self._optimizer.step()
        self._update_t +=1
        
        self._loss_t = loss.detach().cpu().item()
    
    def _calculate_loss(self,transitions:Transition)->torch.Tensor:
        state_t_minus_1 = torch.from_numpy(transitions.state_t_minus_1).to(device=self._device,dtype=torch.float32)
        action_t_minus_1 = torch.from_numpy(transitions.action_t_minus_1).to(device=self._device,dtype=torch.float32)
        reward = torch.from_numpy(transitions.reward_t).to(device=self._device,dtype=torch.float32)
        state_t = torch.from_numpy(transitions.state_t).to(device=self._device,dtype=torch.float32)
        done = torch.from_numpy(transitions.done).to(device=self._device,dtype=torch.bool)
        
        discount_t = (~done).float() * self._discount
        
        q_t_minus_1 = self._main_network(state_t_minus_1).q_values
        with torch.no_grad():
            q_dagger_t = self._target_network(state_t).q_values
        
        loss = loss_lib.q_learning_loss(q_t_minus_1,action_t_minus_1,reward,discount_t,q_dagger_t).loss
        loss = torch.mean(loss)
        # logger.info("loss:" + str(loss))
        return loss
    
    def _update_target_network(self):
        self._target_network.load_state_dict(self._main_network.state_dict())
        self._target_update_t +=1
    def reset(self)->None:
        self._transaction_accumulator.reset()
    @property
    def exploration_rate(self) -> float:
        return self._exploration_rate_lambda(self._step_t)
    
    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            'loss': self._loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
            'exploration_epsilon': self.exploration_rate,
        }


        
