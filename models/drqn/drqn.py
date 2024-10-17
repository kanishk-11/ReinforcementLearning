import copy
from typing import Callable
from ale_py import Action
import numpy as np
import torch
from main.ReplayUtils import transition
from main.ReplayUtils.replayBuffers import PrioritizedReplay, UniformReplay
from main.ReplayUtils.transactionAccumulators import TransitionAccumulator
from main.ReplayUtils.transition import Transition
from main.ReplayUtils.unroller import Unroller
from main.abstractClasses.agent import Agent
from main.abstractClasses.timestep import TimeStepPair
from main.loss.losses import double_q_learning_loss


class DRQN(Agent):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        random_state: np.random.RandomState,
        replay_buffer: UniformReplay,
        transition_accumulator: TransitionAccumulator,
        exploration_rate_lambda: Callable[[int], float],
        learning_interval: int,
        target_network_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        action_space_dimension: int,
        discount: float,
        unroll_length: int,
        clip_gradient: bool,
        max_gradient_norm: float,
        device: torch.device
    ):
        if not 1 <= learning_interval:
            raise ValueError(f'Expect learn_interval to be positive integer, got {learning_interval}')
        if not 1 <= target_network_update_interval:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_network_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be positive integer, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size [1, 512], got {batch_size}')
        if not batch_size <= min_replay_size <= replay_buffer.size:
            raise ValueError(f'Expect min_replay_size >= {batch_size} and <= {replay_buffer.size} and, got {min_replay_size}')
        if not 0 < action_space_dimension:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_space_dimension}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount [0.0, 1.0], got {discount}')
        if not 0 < unroll_length:
            raise ValueError(f'Expect unroll_length to be positive integer, got {unroll_length}')
        self._setup_dqn_meta(device,random_state)
        self._setup_dqn_networks_and_optimizer(network,optimizer)
        self._setup_replay(transition_accumulator,batch_size,replay_buffer,unroll_length)
        self._setup_learning_parameter(discount,exploration_rate_lambda,min_replay_size,learning_interval,target_network_update_interval,clip_gradient,max_gradient_norm)
        # Setup for DQN itself
        self._action_space_dim = action_space_dimension
        self._lstm_state = None  

        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def _setup_dqn_meta(self,device:torch.device,random_state:np.random.RandomState):
        self.agent_name = "DRQN"
        self._device = device
        self._random_state = random_state
    def _setup_dqn_networks_and_optimizer(self,network:torch.nn.Module,optimizer:torch.optim.Optimizer):
        self._main_network = network.to(self._device)
        self._target_network = copy.deepcopy(self._main_network).to(self._device)
        self._optimizer = optimizer
        for p in self._target_network.parameters():
            p.requires_grad = False
    def _setup_replay(self,transition_accumulator:TransitionAccumulator,batch_size:int,replay_buffer:UniformReplay,unroll_length:int):
        self._transaction_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._unroller = Unroller(
            unroll_length = unroll_length,
            overlap = 1 ,
            structure = Transition(None, None,None,None,None),
            cross_episode = True
        )
        self._replay_buffer = replay_buffer
    def _setup_learning_parameter(self,discount:float,exploration_rate_lambda:Callable[[int],float],min_replay_size:int,learning_interval:int,target_network_update_interval:int,clip_gradient:bool,max_gradient_norm:float):
        self._exploration_rate_lambda = exploration_rate_lambda
        self._clip_gradient = clip_gradient
        self._discount = discount
        self._min_replay_size = min_replay_size
        self._learning_interval = learning_interval
        self._target_network_update_interval = target_network_update_interval
        self._max_gradient_norm = max_gradient_norm
    
    def step(self,timestep:TimeStepPair)->Action:
        self._step_t +=1
        action_t = self.act(timestep)
        for transition in self._transaction_accumulator.step(timestep,action_t):
            unrolled_tranition =  self._unroller.add(transition,timestep.done)
            if unrolled_tranition is not None:
                self._replay_buffer.add(unrolled_tranition)
        if self._replay_buffer.transitions_added < self._min_replay_size:
            return action_t
        if self._step_t % self._learning_interval == 0:
            self.learn()
        return action_t
    def reset(self):
        self._transaction_accumulator.reset()
        self._lstm_state = self._main_network.get_initial_hidden_state(batch_size=1)
    
    def act(self,timestep:TimeStepPair)->Action:
        action_t = self._choose_action(timestep,self.exploration_rate)
        return action_t
    @torch.no_grad()
    def _choose_action(self,timestep:TimeStepPair,exploration_rate:float)->Action:
        if self._random_state.rand() <= exploration_rate:
            action_t = self._random_state.randint(0,self._action_space_dim)
            return action_t
        state_t = torch.from_numpy(timestep.observation[None,None,...]).to(device=self._device,dtype=torch.float32)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        out = self._main_network(state_t,hidden_state)
        q_values = out.q_values
        action_t = torch.argmax(q_values,dim=-1)
        return action_t.cpu().item()
    def learn(self):
        transitions = self._replay_buffer.sample(self._batch_size)
        self._update(transitions)
        if self._update_t > 1 and self._update_t % self._target_network_update_interval == 0:
            self._update_target_network()
    
    def _update(self,transitions):
        self._optimizer.zero_grad(set_to_none=True)
        loss = self._calculate_loss(transitions)
        loss.backward()
        if self._clip_gradient:
            torch.nn.utils.clip_grad_norm_(self._main_network.parameters(), self._max_gradient_norm, error_if_nonfinite=True)
        self._optimizer.step()
        self._update_t +=1
        self._loss_t = loss.detach().cpu().item()
    
    def _calculate_loss(self,transitions:Transition):
        state_t =  torch.from_numpy(transitions.state_t).to(device=self._device,dtype=torch.float32)
        state_t_minus_1 =  torch.from_numpy(transitions.state_t_minus_1).to(device=self._device,dtype=torch.float32)
        reward =  torch.from_numpy(transitions.reward_t).to(device=self._device,dtype=torch.float32)
        action_t_minus_1 =  torch.from_numpy(transitions.action_t_minus_1).to(device=self._device,dtype=torch.int64)
        done =  torch.from_numpy(transitions.done).to(device=self._device,dtype=torch.bool)
        hidden_state = self._main_network.get_initial_hidden_state(batch_size = self._batch_size)
        hidden_state = tuple(s.to(device=self._device) for s in hidden_state)
        target_hidden_state = tuple(hx.clone().to(device=self._device) for hx in hidden_state)
        discount = (~done).float() * self._discount
        q_t_minus_1 = self._main_network(state_t_minus_1,hidden_state).q_values
        with torch.no_grad():
            q_t_selector = self._main_network(state_t,hidden_state).q_values
            target_q_t = self._target_network(state_t,target_hidden_state).q_values  
        B,T = state_t_minus_1.shape[:2]
        q_t_minus_1 = q_t_minus_1.view(B*T,-1)
        action_t_minus_1 = action_t_minus_1.view(B*T)
        reward = reward.view(B*T)
        discount = discount.view(B*T)
        target_q_t = target_q_t.view(B*T,-1)
        q_t_selector = target_q_t.view(B*T,-1)
        
        loss = double_q_learning_loss(q_t_minus_1,action_t_minus_1,reward,discount,target_q_t,q_t_selector).loss
        loss = torch.mean(loss,dim=0)
        return loss
    def _update_target_network(self):
        self._target_network.load_state_dict(self._main_network.state_dict())
        self._target_update_t += 1
    @property
    def exploration_rate(self):
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