

import copy
import multiprocessing
from typing import Iterable, Mapping, NamedTuple, Optional, Text, Tuple
from ale_py import Action
import numpy as np
from sympy import bool_map
import torch

from main.ReplayUtils.common import split_from_structure
from main.ReplayUtils.replayBuffers import PrioritizedReplay
from main.ReplayUtils.unroller import Unroller
from main.abstractClasses import action
from main.abstractClasses.agent import Agent
from main.abstractClasses.learner import Learner
from main.abstractClasses.timestep import TimeStepPair
from main.functions import calculate_distributed_priorities_from_td_error, get_actors_exploration_rate, n_step_bellman_target
from main.networks.value_networks import RNNDQNOutputs
from main.transforms import signed_hyperbolic, signed_parabolic
from models.common import disable_autograd, numpy_to_tensor


HiddenState = Tuple[torch.Tensor,torch.Tensor]
class R2D2Transition(NamedTuple):
    state_t : Optional[np.ndarray]
    reward : Optional[float]
    done : Optional[bool]
    action_t : Optional[int]
    q_values: Optional[np.ndarray]
    last_action: Optional[int]
    initital_hidden_state: Optional[np.ndarray]
    initial_cell_state: Optional[np.ndarray]
def calculate_loss_and_priorities(
    q_values:torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    target_q_values: torch.Tensor,
    target_action: torch.Tensor,
    discount: float,
    n_step :int,
    epsilon: float=0.001,
    eta: float=0.9
) -> Tuple[torch.Tensor,torch.Tensor]:
    q_values = q_values.gather(-1,action[...,None]).squeeze(-1)
    target_q_max = target_q_values.gather(-1,target_action[...,None]).squeeze(-1)
    target_q_max = signed_parabolic(target_q_max,epsilon)
    target_q = n_step_bellman_target(reward,done,target_q_max,discount,n_step)
    q_values = q_values[:-1,...]
    target_q = target_q[1:, ...]
    target_q = signed_hyperbolic(target_q,epsilon)
    if q_values.shape != target_q.shape:
        raise ValueError("q_values and target_q dont have the same shape")
    td_error = target_q - q_values
    with torch.no_grad():
        priorities = calculate_distributed_priorities_from_td_error(td_error,eta)
    losses = 0.5*(torch.sum(torch.square(td_error),dim=0))
    return losses, priorities

class Actor(Agent):
    def __init__(
        self,
        rank:int,
        data_queue: multiprocessing.Queue,
        network: torch.nn.Module,
        random_state:np.random.RandomState,
        num_actors:int,
        action_dim: int,
        unroll_length:int,
        burn_in:int,
        actor_update_interval:int,
        device:int,
        shared_parameters:dict
    ):
        if num_actors<1:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}') 
        if action_dim<1:
            raise ValueError(f'Expected action_dim to be positive integer, got {action_dim}')
        if unroll_length < 1:
            raise ValueError(f'Expect unroll_length to be positive integer, got {unroll_length}')
        if not 0<= burn_in<unroll_length:
            raise ValueError(f'Expect 0<= burn_in < unroll_length, got {burn_in}')
        if not 1<= actor_update_interval:
            raise ValueError(f'Expect actor_update_interval to be positive integer, got {actor_update_interval}')
        self.rank = rank
        self.agent_name = f'R2D2-Actor_Rank{rank}'
        self._network = network
        disable_autograd(self._network)
        self._shared_params = shared_parameters
        self._queue = data_queue
        self._device = device
        self._random_state = random_state
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval
        self._unroller = Unroller(
            unroll_length=unroll_length,
            overlap=burn_in+1,
            structure=R2D2Transition(None,None,None,None,None,None,None,None),
            cross_episode=False
        )
        epsilons = get_actors_exploration_rate(num_actors)
        self._exploration_rate = epsilons[self.rank]
        self._last_action = None
        self._lstm_state = None
        self._step_t = -1
    
    @torch.no_grad()
    def step(self,timestep:TimeStepPair)->Action:
        self._step_t +=1
        if self._step_t % self._actor_update_interval == 0:
            self._update_actor_network()
        q_values, action, hidden_state = self.act(timestep)
        transition= R2D2Transition(
            state_t=timestep.observation,
            action_t=action,
            q_values=q_values,
            reward=timestep.reward,
            done=timestep.done,
            last_action=self._last_action,
            initital_hidden_state=hidden_state[0].squeeze(1).cpu().numpy(),
            initial_cell_state=hidden_state[1].squeeze(1).cpu().numpy()
        )
        unrolled_transition = self._unroller.add(transition,timestep.done)
        self._last_action, self._lstm_state = action,hidden_state
        if unrolled_transition is not None:
            self._load_unrolled_to_queue(unrolled_transition)
        return action
    
    def reset(self):
        self._unroller.reset()
        self._last_action = self._random_state.randint(0, self._action_dim)  # Initialize a_tm1 randomly
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
    def act(self,timestep:TimeStepPair)->Tuple[np.ndarray,Action,Tuple[torch.Tensor]]:
        return self._choose_action(timestep,self._exploration_rate)
    
    @torch.no_grad()
    def _choose_action(self,timestep:TimeStepPair,exploration_rate:float):
        state_t,action_t_minus_1,reward,hidden_state = self._prepare_input(timestep)
        pi_output = self._network(state_t,action_t_minus_1,reward,hidden_state)
        q_values = pi_output.q_values.squeeze()
        action = torch.argmax(q_values,dim=-1).cpu().item()
        
        if self._random_state.rand() <= exploration_rate:
            action = self._random_state.randint(0,self._action_dim)
        return q_values.cpu().numpy(), action, pi_output.hidden_state
    
    def _prepare_input(self,timestep:TimeStepPair):
        state_t = torch.from_numpy(timestep.observation[None,...]).to(self._device,dtype=torch.float32)
        action_t_minus_1 = torch.tensor(self._last_action,device=self._device,dtype=torch.int64)
        reward = torch.tensor(timestep.reward,device=self._device,dtype=torch.float32)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        return state_t.unsqueeze(0), action_t_minus_1.unsqueeze(0),reward.unsqueeze(0),hidden_state
    def _load_unrolled_to_queue(self,unrolled_transitions):
        self._queue.put(unrolled_transitions)
    def _update_actor_network(self):
        state_dict = self._shared_params['network']
        if state_dict is not None:
            if self._device != 'cpu':
                state_dict = {k: v.to(device=self._device) for k, v in state_dict.items()}
            self._network.load_state_dict(state_dict)
    @property
    def statistics(self) -> Mapping[str, float]:
        return {'exploration_epsilon': self._exploration_rate}

class R2D2Learner(Learner):
    def __init__(
        self,
        network:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        replay_buffer:PrioritizedReplay,
        target_network_update_interval:int,
        min_replay_size:int,
        batch_size:int,
        n_steps:int,
        discount:float,
        burn_in:int,
        priority_eta:float,
        rescale_epsilon:float,
        clip_gradient:bool,
        max_gradient_norm:float,
        device:torch.device,
        shared_parameters:dict
    ):
        if target_network_update_interval < 1:
            raise ValueError(f'Expect target_net_update_interval to be positive integer, got {target_network_update_interval}')
        if not 1 <= min_replay_size:
            raise ValueError(f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}')
        if not 1 <= batch_size <= 512:
            raise ValueError(f'Expect batch_size to in the range [1, 512], got {batch_size}')
        if not 1 <= n_steps:
            raise ValueError(f'Expect n_step to be integer greater than or equal to 1, got {n_steps}')
        if not 0.0 <= discount <= 1.0:
            raise ValueError(f'Expect discount to in the range [0.0, 1.0], got {discount}')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}')
        if not 0.0 <= rescale_epsilon <= 1.0:
            raise ValueError(f'Expect rescale_epsilon to in the range [0.0, 1.0], got {rescale_epsilon}')
        self.agent_name = 'R2D2-Learner'
        self._device = device
        self._network = network.to(device=device)
        self._optimizer = optimizer
        self._target_network = copy.deepcopy(self._network).to(device=self._device)
        disable_autograd(self._target_network)
        self._shared_parameters = shared_parameters
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._burn_in = burn_in
        self._target_network_update_interval = target_network_update_interval
        self._discount = discount
        self._clip_gradient = clip_gradient
        self._max_gradient_norm = max_gradient_norm
        self._rescale_epsilon = rescale_epsilon
        
        self._replay = replay_buffer
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_seen_priority = 1.0  # New unroll will use this as priority

        # Counters
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan
        
    def step(self) -> Iterable[Mapping[Text, float]]:
        self._step_t += 1

        if self._replay.transitions_added < self._min_replay_size or self._step_t % max(4, int(self._batch_size * 0.25)) != 0:
            return

        self._learn()
        yield self.statistics
    def reset(self) -> None:
        return
        """Called for reset"""
    def received_item_from_queue(self, item) -> None:
        """Received item send by actors through multiprocessing queue."""
        self._replay.add(item, self._max_seen_priority)
    
    def get_network_state_dict(self):
        # To keep things consistent, we move the parameters to CPU
        return {k: v.cpu() for k, v in self._network.state_dict().items()}
    def _learn(self)->None:
        transitions,indices,weights = self._replay.sample(self._batch_size)
        priorities = self._update(transitions,weights)
        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape ({self._batch_size},), got {priorities.shape}')
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority,np.max(priorities)])
        self._replay.update_priorities(indices,priorities)
        self._shared_parameters['network'] = self.get_network_state_dict()

        # Copy online Q network parameters to target Q network, every m updates
        if self._update_t > 1 and self._update_t % self._target_network_update_interval == 0:
            self._update_target_network()
    def _update(self,transitions:R2D2Transition,weights):
        weights = torch.from_numpy(weights).to(device=self._device, dtype=torch.float32)  # [batch_size]
        initial_hidden_state = self._get_initial_hidden_state(transitions)
        burn_transitions , learn_tranisitions = split_from_structure(transitions,R2D2Transition(None,None,None,None,None,None,None,None),self._burn_in)
        if burn_transitions is not None:
            hidden_state, target_hidden_state = self._burn_in_unrolled(burn_transitions,initial_hidden_state)
        else:
            hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
            target_hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
        self._optimizer.zero_grad(set_to_none=True)
        loss,priorities = self._calculate_loss(learn_tranisitions,hidden_state,target_hidden_state)
        
        loss = torch.mean(loss* weights.detach())
        loss.backward()
        if self._clip_gradient:
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._max_gradient_norm)
        self._optimizer.step()
        self._loss_t = loss.detach().cpu().item()
        self._update_t+=1

        return priorities
    def _calculate_loss(self,transitions:R2D2Transition,hidden_state:HiddenState,target_hidden_state:HiddenState):
        state_t = torch.from_numpy(transitions.state_t).to(device=self._device, dtype=torch.float32)
        action_t = torch.from_numpy(transitions.action_t).to(device=self._device, dtype=torch.int64)
        last_action = torch.from_numpy(transitions.last_action).to(device=self._device, dtype=torch.int64)
        reward = torch.from_numpy(transitions.reward).to(device=self._device, dtype=torch.float32)
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)
        q_values = self._network(state_t,last_action,reward,hidden_state).q_values
        with torch.no_grad():
            best_action = torch.argmax(q_values, dim=-1)
            target_q_values = self._target_network(state_t,last_action,reward,target_hidden_state).q_values
        losses,priorities = calculate_loss_and_priorities(q_values,action_t,reward,done,target_q_values,best_action,self._discount,self._n_steps,self._rescale_epsilon,self._priority_eta)
        return (losses, priorities)
    
    @torch.no_grad()
    def _burn_in_unrolled(self,burn_transitions:R2D2Transition,initial_hidden_state:HiddenState):
        state_t = torch.from_numpy(burn_transitions.state_t).to(device=self._device,dtype=torch.float32)
        last_action = torch.from_numpy(burn_transitions.action_t).to(device=self._device,dtype=torch.int64)
        reward = torch.from_numpy(burn_transitions.reward).to(device=self._device,dtype=torch.float32)
        
        hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
        target_hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
        final_hidden_state =  self._network(state_t,last_action,reward,hidden_state).hidden_state
        final_target_hidden_state = self._target_network(state_t,last_action,reward,target_hidden_state).hidden_state
        return final_hidden_state, final_target_hidden_state
    def _get_initial_hidden_state(self,transitions:R2D2Transition)->Tuple[torch.Tensor,torch.Tensor]:
        init_h = torch.from_numpy(transitions.initital_hidden_state[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        init_c = torch.from_numpy(transitions.initial_cell_state[0:1]).squeeze(0).to(device=self._device, dtype=torch.float32)
        init_h = init_h.swapaxes(0, 1)
        init_c = init_c.swapaxes(0, 1)
        return (init_h, init_c)
    def _update_target_network(self):
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_update_t += 1

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'loss': self._loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }