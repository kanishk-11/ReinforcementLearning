





import copy
import logging
import multiprocessing
from re import S
import time
from typing import Iterable, Mapping, NamedTuple, Optional, Text, Tuple
import numpy as np
import torch

from main.ReplayUtils.common import split_from_structure
from main.ReplayUtils.replayBuffers import PrioritizedReplay
from main.ReplayUtils.unroller import Unroller
from main.abstractClasses import timestep
from main.abstractClasses.action import Action
from main.abstractClasses.agent import Agent
from main.abstractClasses.learner import Learner
from main.abstractClasses.timestep import TimeStepPair
from main.curiosity.episodic_bonus import EpisodicBonusModule
from main.curiosity.lifelong_bonus import RNDLifeLongBonusModule
from main.functions import calculate_distributed_priorities_from_td_error, get_actors_exploration_rate, get_ngu_policy_betas_and_gammas, transformed_retrace
from main.networks.value_networks import NGUNetworkInputs
from main.normalizers import PytorchRunningMeanStd
from main.transforms import IDENTITY_PAIR, SIGNED_HYPERBOLIC_PAIR
from models.common import disable_autograd, numpy_to_tensor


torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]

class NGUTransition(NamedTuple):
    state_t : Optional[np.ndarray]
    action_t : Optional[np.ndarray]
    q_values_t : Optional[np.ndarray]
    action_probability_t : Optional[np.ndarray]
    last_action : Optional[np.ndarray]
    extrinsic_reward_t : Optional[float]
    intrinsic_reward_t : Optional[float]
    policy_index: Optional[int]
    beta : Optional[float]
    discount : Optional[float]
    done : Optional[bool]
    init_h : Optional[np.ndarray]
    init_c : Optional[np.ndarray]

Transition = NGUTransition(None,None,None,None,None,None,None,None,None,None,None,None,None)

class NGUActor(Agent):
    def __init__(
        self,
        rank:int,
        data_queue: multiprocessing.Queue,
        network: torch.nn.Module,
        RND_predictor_network: torch.nn.Module,
        RND_target_network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        random_state:np.random.RandomState,
        extrinsic_discount:float,
        intrinsic_discount:float,
        num_actors:int,
        action_dim:int,
        unroll_length:int,
        burn_in:int,
        num_policies:int,
        policy_beta : float,
        episodic_memory_size: int,
        reset_episodic_memory:bool,
        num_neighbors:int,
        cluster_distance:float,
        kernel_epsilon:float,
        maximum_similarity:float,
        actor_update_interval:int,
        device:torch.device,
        shared_parameters,
        c_constant:float=0.001
    ) -> None:
        if not 0.0 <= extrinsic_discount <= 1.0:
            raise ValueError(f'Expect extrinsic_discount to be in [0,1], got {extrinsic_discount}')
        if not 0.0 <= intrinsic_discount <= 1.0:
            raise ValueError(f'Expect intrinsic_discount to be in [0,1], got {intrinsic_discount}')
        if not num_actors > 0:
            raise ValueError(f'Expect num_actors to be positive integer, got {num_actors}')
        if not action_dim > 0:
            raise ValueError(f'Expect action_dim to be positive integer, got {action_dim}')
        if not unroll_length > 0:
            raise ValueError(f'Expect unroll_length to be positive integer, got {unroll_length}')
        if not 0 <= burn_in < unroll_length:
            raise ValueError(f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}')
        if not num_policies>0:
            raise ValueError(f'Expect num_policies to be positive integer, got {num_policies}')
        if not 0.0 <= policy_beta <= 1.0:
            raise ValueError(f'Expect policy_beta to be in [0,1], got {policy_beta}')
        if not episodic_memory_size > 0:
            raise ValueError(f'Expect episodic_memory_size to be positive integer, got {episodic_memory_size}')
        if not num_neighbors > 0 :
            raise ValueError(f'Expect num_neighbours to be positive integer, got {num_neighbors}')
        if not 0.0 <= cluster_distance <= 1.0:
            raise ValueError(f'Expect cluster_distance to be in [0,1], got {cluster_distance}')
        if not 0.0 <= kernel_epsilon <= 1.0:
            raise ValueError(f'Expect kernel_epsilon to be in [0,1], got {kernel_epsilon}')
        if not actor_update_interval > 0:
            raise ValueError(f'Expect actor_update_interval to be positive integer, got {actor_update_interval}')
        
        self.rank = rank
        self.agent_name = f'NGU-Agent-Rank-{rank}'
        
        self._network = network.to(device=device)
        self._rnd_target_network = RND_target_network.to(device=device)
        self._rnd_predictor_network = RND_predictor_network.to(device=device)
        self._embedding_network = embedding_network.to(device=device)
        
        disable_autograd(self._network)
        disable_autograd(self._rnd_target_network)
        disable_autograd(self._rnd_predictor_network)
        disable_autograd(self._embedding_network)
        self._shared_parameters = shared_parameters
        
        self._data_queue = data_queue
        self._device = device
        self._random_state = random_state
        self._num_actors = num_actors
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval
        self._num_policies = num_policies
        self._unroller = Unroller(
            unroll_length=unroll_length,
            overlap=burn_in+1,
            structure=NGUTransition(None,None,None,None,None,None,None,None,None,None,None,None,None),
            cross_episode=False
        )
        self._betas ,self._gammas = get_ngu_policy_betas_and_gammas(
            num_policies=num_policies,
            beta = policy_beta,
            gamma_maximum = extrinsic_discount,
            gamma_minimum = intrinsic_discount
        )
        self._policy_index = None
        self._policy_beta = None
        self._policy_discount = None
        self._sample_policy()
        self._reset_episodic_memory = reset_episodic_memory
        self._exploration_rate = get_actors_exploration_rate(n=num_actors)[self.rank]
        
        self._episodic_bonus_module = EpisodicBonusModule(
            embedding_network = embedding_network,
            device = device,
            size = episodic_memory_size,
            num_neighbors = num_neighbors,
            kernel_epsilon = kernel_epsilon,
            cluster_distance = cluster_distance,
            maximum_similarity = maximum_similarity,
            c_constant=c_constant,
        )
        self._life_long_bonus_module = RNDLifeLongBonusModule(
            target_network = RND_target_network,
            predictor_network = RND_predictor_network,
            device = device,
            discount = intrinsic_discount
        )
        self._last_action = None
        self._episodic_bonus_t = None
        self._life_long_bonus_t = None
        self._lstm_state = None
        self._step_t = -1
        self._embedding_update_count = 0
        self._rnd_update_count = 0
        self._actor_update_count=0
        self._q_update_timestamp = None
        self._rnd_update_timestamp = None
        self._embedding_update_timestamp = None
    @torch.no_grad()
    def step(self, timestep_pair: TimeStepPair) -> Action:
        self._step_t += 1
        if self._step_t % self._actor_update_interval ==0 :
            self._update_actor_network(False)
        q_values, action,action_probability, hidden_state = self.act(timestep_pair)
        transition = NGUTransition(
            state_t=timestep_pair.observation,
            action_t=action,
            q_values_t=q_values,
            action_probability_t=action_probability,
            last_action=self._last_action,
            extrinsic_reward_t=timestep_pair.reward,
            intrinsic_reward_t=self.intrinsic_reward,
            policy_index=self._policy_index,
            beta = self._policy_beta,
            discount=self._policy_discount,
            done=timestep_pair.done,
            init_h=self._lstm_state[0].squeeze(1).cpu().numpy(),
            init_c=self._lstm_state[1].squeeze(1).cpu().numpy(),
        )
        unrolled_transition = self._unroller.add(transition,timestep_pair.done)
        state = numpy_to_tensor(timestep_pair.observation[None,...],self._device,torch.float32)
        llb = self._life_long_bonus_module.get_bonus(state)
        eb = self._episodic_bonus_module.get_bonus(state)
        # print(llb,eb)
        self._life_long_bonus_t = llb
        self._episodic_bonus_t = eb
        self._last_action = action
        self._lstm_state = hidden_state
        if unrolled_transition is not None:
            self._load_unrolled_transition_to_queue(unrolled_transition)
        return action

    def reset(self):
        self._unroller.reset()
        if self._reset_episodic_memory:
            self._episodic_bonus_module.reset()
        self._update_actor_network(True)
        self._sample_policy()
        self._last_action = self._random_state.randint(0, self._action_dim)
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
    def act(self,timestep:TimeStepPair):
        return self._choose_action(timestep)
    @torch.no_grad()
    def _choose_action(self,timestep:TimeStepPair):
        pi = self._network(self._network_inputs(timestep))
        q_values = pi.q_values.squeeze()
        action = torch.argmax(q_values,dim=-1).cpu().item()
        action_probability = 1 - (self._exploration_rate*((self._action_dim-1)/self._action_dim))
        if self._random_state.rand() < self._exploration_rate:
            action = self._random_state.randint(0,self._action_dim)
            action_probability = self._exploration_rate/self._action_dim
        
        return q_values.cpu().numpy() , action,action_probability,pi.hidden_state
    def _network_inputs(self,timestep:TimeStepPair):
        state = numpy_to_tensor(timestep.observation[None,...],self._device,torch.float32)
        last_action = torch.tensor(self._last_action).to(device= self._device,dtype=torch.int64)
        extrinsic_reward = torch.tensor(timestep.reward).to(device=self._device,dtype=torch.float32)
        intrinsic_reward = torch.tensor(self.intrinsic_reward).to(device=self._device,dtype=torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device,dtype=torch.int64)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        return NGUNetworkInputs(
            state_t = state.unsqueeze(0),
            action_t_minus_1 = last_action.unsqueeze(0),
            extrinsic_reward_t = extrinsic_reward.unsqueeze(0),
            intrinsic_reward_t = intrinsic_reward.unsqueeze(0),
            policy_index_t = policy_index.unsqueeze(0),
            hidden_state = hidden_state,
        )
    def _load_unrolled_transition_to_queue(self,transition:NGUTransition):
        self._data_queue.put(transition)
    def _update_actor_network(self,update_embedding:bool = False):
        # print("_update_actor_network called with update_embedding" + str(update_embedding))
        q_state_dict = self._shared_parameters['network']
        embedding_state_dict = self._shared_parameters['embedding_network']
        RND_state_dict = self._shared_parameters['rnd_predictor_network']
        weight_update_timestamp = self._shared_parameters['timestamp']
        # logging.info("RND_state_dict={}".format(RND_state_dict))
        # logging.info("embedding_state_dict={}".format(embedding_state_dict))
        # logging.info("q_state_dict={}".format(q_state_dict))
        
        if RND_state_dict is not None and self._rnd_update_timestamp !=weight_update_timestamp:
            if self._device != 'cpu':
                RND_state_dict = {k: v.to(device=self._device) for k, v in RND_state_dict.items()}
            self._rnd_predictor_network.load_state_dict(RND_state_dict)
            self._rnd_update_count+=1
            self._rnd_update_timestamp =weight_update_timestamp

        if embedding_state_dict is not None and update_embedding==True and self._embedding_update_timestamp != weight_update_timestamp:
            if self._device != 'cpu':
                embedding_state_dict = {k: v.to(device=self._device) for k, v in embedding_state_dict.items()}
            self._embedding_network.load_state_dict(embedding_state_dict)
            self._embedding_update_count+=1
            self._embedding_update_timestamp =weight_update_timestamp

            
        if q_state_dict is not None and self._q_update_timestamp != weight_update_timestamp:
            if self._device!= 'cpu':
                q_state_dict = {k: v.to(device=self._device) for k, v in q_state_dict.items()}
            self._network.load_state_dict(q_state_dict)
            self._actor_update_count+=1
            self._q_update_timestamp =weight_update_timestamp

        self._episodic_bonus_module.update_embedding_network(self._embedding_network.state_dict())
        self._life_long_bonus_module.update_predictor_network(self._rnd_predictor_network.state_dict())
    def _sample_policy(self):
        self._policy_index = np.random.randint(0, self._num_policies)
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]
    @property
    def intrinsic_reward(self):
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t,1.0),5.0)
    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current actor's statistics as a dictionary."""
        return {
            'policy_discount': self._policy_discount,
            'policy_beta': self._policy_beta,
            'exploration_epsilon': self._exploration_rate,
            'intrinsic_reward': self.intrinsic_reward,
            'episodic_bonus': self._episodic_bonus_t,
            'lifelong_bonus': self._lifelong_bonus_t,
            'embedding_update_count': self._embedding_update_count,
            'rnd_update_count':self._actor_update_count,
            'actor_update_count':self._actor_update_count,
        }
        
class NGULearner(Learner): 
    def __init__(
        self,
        network:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        embedding_network:torch.nn.Module,
        RND_target_network:torch.nn.Module,
        RND_predictor_network:torch.nn.Module,
        intrinsic_embedding_optimizer:torch.optim.Optimizer,
        intrinsic_rnd_optimizer:torch.optim.Optimizer,
        replay_buffer: PrioritizedReplay,
        target_network_update_interval:int,
        minimum_replay_size:int,
        batch_size:int,
        unroll_length:int,
        burn_in:int,
        retrace_lambda:float,
        transformed_retrace:bool,
        priority_eta:float,
        clip_gradients:bool,
        max_gradient_norm:float,
        device:torch.device,
        shared_parameters
    ):
        if not target_network_update_interval > 0:
            raise ValueError('target_network_update_interval must be > 0')
        if not minimum_replay_size > 0:
            raise ValueError('minimum_replay_size must be > 0')
        if not batch_size > 0:
            raise ValueError('batch_size must be > 0')
        if not unroll_length > 0:
            raise ValueError('unroll_length must be > 0')
        if not burn_in > 0:
            raise ValueError('burn_in must be > 0')
        if not 0.0 <= retrace_lambda <= 1.0:
            raise ValueError('retrace_lambda must be in [0,1]')
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError('priority_eta must be in [0,1]')
        if not 0.0 <= burn_in <= unroll_length:
            raise ValueError('burn_in must be in [0,unroll_length]')
        self.agent_name = f'NGU-Learner'
        self._device = device
        self._network = network.to(device=device)
        self._network.train()
        self._optimizer = optimizer
        self._embedding_network = embedding_network.to(device=device)
        self._embedding_network.train()
        self._rnd_predictor_network = RND_predictor_network.to(device=device)
        self._rnd_predictor_network.train()
        self._intrinsic_embedding_optimizer = intrinsic_embedding_optimizer
        self._intrinsic_rnd_optimizer = intrinsic_rnd_optimizer

        self._rnd_target_network = RND_target_network.to(device=device)
        self._target_network = copy.deepcopy(self._network).to(device=device)
        disable_autograd(self._target_network)
        disable_autograd(self._rnd_target_network)
        self._shared_parameters = shared_parameters
        self._batch_size = batch_size
        self._burn_in = burn_in
        self._unroll_length = unroll_length
        self._total_unroll_length = unroll_length + 1
        self._target_network_update_interval = target_network_update_interval
        self._clip_gradients = clip_gradients
        self._max_gradient_norm = max_gradient_norm
        
        self._rnd_observation_normalizer = PytorchRunningMeanStd(shape = (1,84,84),device = self._device)
        
        self._replay_buffer = replay_buffer
        self._min_replay_size = minimum_replay_size
        self._priority_eta  = priority_eta
        
        self._max_seen_priority = 1.0
        
        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace
        
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._retrace_loss_t = np.nan
        self._rnd_loss_t = np.nan
        self._embedding_network_loss_t = np.nan
    
    def step(self) -> Iterable[Mapping[Text,float]]:
        self._step_t +=1 
        if self._replay_buffer.transitions_added < self._min_replay_size or self._step_t % max(4,int(self._batch_size*0.25)) !=0:
            return
        self._learn()
        yield self.statistics
    def reset(self) -> None:
        return
    def received_item_from_queue(self, item) -> None:
        self._replay_buffer.add(item,self._max_seen_priority)
    
    def get_network_state_dict(self):
        return {k:v.cpu() for k,v in self._network.state_dict().items()}
    def get_embedding_network_state_dict(self):
        return {k:v.cpu() for k,v in self._embedding_network.state_dict().items()}
    def get_rnd_predictor_network_state_dict(self):
        return {k:v.cpu() for k,v in self._rnd_predictor_network.state_dict().items()}
    def _learn(self):
        transitions , indices, weights = self._replay_buffer.sample(self._batch_size)
        priorities = self._update_q_network(transitions,weights)
        self._update_embedding_and_rnd_predictor(transitions,weights)
        self._update_t +=1
        if priorities.shape != (self._batch_size,):
            raise ValueError("priorities has shape mismatch")
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority,np.max(priorities)])
        self._replay_buffer.update_priorities(indices,priorities)
        self._shared_parameters['network'] = self.get_network_state_dict()
        self._shared_parameters['embedding_network'] = self.get_embedding_network_state_dict()
        self._shared_parameters['rnd_predictor_network'] = self.get_rnd_predictor_network_state_dict()
        self._shared_parameters['timestamp'] = time.time_ns()
        if self._update_t > 1 and self._update_t % self._target_network_update_interval == 0:
            self._update_target_network()
    def _update_q_network(self,transitions:NGUTransition,weights:np.ndarray):
        weights = numpy_to_tensor(weights,self._device,torch.float32)
        init_hidden_state = self._extract_hidden_state(transitions)
        burn_in_transitions,learn_transitions = split_from_structure(transitions,Transition,self._burn_in)
        if burn_in_transitions is not None:
            hidden_state ,target_hidden_state = self._burn_in_unrolled_q_network(burn_in_transitions,init_hidden_state)
        else:
            hidden_state = tuple(s.clone().to(device=self._device) for s in init_hidden_state)
            target_hidden_state = tuple(s.clone().to(device=self._device) for s in init_hidden_state)
        self._optimizer.zero_grad()
        
        q_values = self._get_predicted_q_values(learn_transitions,self._network,hidden_state)
        with torch.no_grad():
            target_q_values = self._get_predicted_q_values(learn_transitions,self._target_network,target_hidden_state)
        retrace_loss , priorities = self._calculate_retrace_loss(learn_transitions,q_values,target_q_values.detach())
        loss = torch.mean(retrace_loss * weights.detach())
        loss.backward()
        
        if self._clip_gradients:
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._max_gradient_norm)
        self._optimizer.step()

        # For logging only.
        self._retrace_loss_t = loss.detach().cpu().item()

        return priorities
    def _get_predicted_q_values(self,transitions:NGUTransition,q_network:torch.nn.Module,hidden_state:HiddenState)->torch.Tensor:
        state_t = numpy_to_tensor(transitions.state_t,self._device,torch.float32)
        last_action =numpy_to_tensor(transitions.last_action,self._device,torch.int64)
        extrinsic_reward = numpy_to_tensor(transitions.extrinsic_reward_t,self._device,torch.float32)
        intrinsic_reward = numpy_to_tensor(transitions.intrinsic_reward_t,self._device,torch.float32)
        policy_index = numpy_to_tensor(transitions.policy_index,self._device,torch.int64)
        
        assert not torch.any(torch.isnan(state_t))
        assert not torch.any(torch.isnan(last_action))
        assert not torch.any(torch.isnan(extrinsic_reward))
        assert not torch.any(torch.isnan(intrinsic_reward))
        assert not torch.any(torch.isnan(policy_index))

        # Get q values from Q network
        q_t = q_network(
            NGUNetworkInputs(
                state_t=state_t,
                action_t_minus_1=last_action,
                extrinsic_reward_t=extrinsic_reward,
                intrinsic_reward_t=intrinsic_reward,
                policy_index_t=policy_index,
                hidden_state=hidden_state,
            )
        ).q_values

        assert not torch.any(torch.isnan(q_t))

        return q_t
    def _calculate_retrace_loss(
        self,
        transitions:NGUTransition,
        q_values:torch.Tensor,
        target_q_values:torch.Tensor,
    )->Tuple[torch.Tensor,np.ndarray]:
        action_t = numpy_to_tensor(transitions.action_t,self._device,torch.int64)
        action_probability = numpy_to_tensor(transitions.action_probability_t,self._device,torch.float32)
        extrinsic_reward_t = numpy_to_tensor(transitions.extrinsic_reward_t,self._device,torch.float32)
        intrinsic_reward_t = numpy_to_tensor(transitions.intrinsic_reward_t,self._device,torch.float32)
        beta = numpy_to_tensor(transitions.beta,self._device,torch.float32)
        discount = numpy_to_tensor(transitions.discount,self._device,torch.float32)
        done = numpy_to_tensor(transitions.done,self._device,torch.bool)
        
        reward_t = extrinsic_reward_t + beta * intrinsic_reward_t
        discount_t = (~done).float() * discount
        target_policy_probabilities = torch.nn.functional.softmax(target_q_values,dim=-1)
        if self._transformed_retrace:
            transformation_pair = SIGNED_HYPERBOLIC_PAIR
        else:
            transformation_pair = IDENTITY_PAIR
        retrace = transformed_retrace(
            q_t_minus_1 = q_values[:-1],
            q_t = target_q_values[1:],
            action_t_minus_1 = action_t[:-1],
            action_t = action_t[1:],
            reward_t = reward_t[1:],
            discount_t = discount_t[1:],
            pi_t = target_policy_probabilities[1:],
            mu_t = action_probability[1:],
            transformation_pair = transformation_pair,
            lambda_ = self._retrace_lambda,
        )
        with torch.no_grad():
            priorities = calculate_distributed_priorities_from_td_error(retrace.extraInformation.td_error,self._priority_eta)
        loss = torch.sum(retrace.loss , dim =0)
        return loss, priorities
    def _update_embedding_and_rnd_predictor(self,transitions:NGUTransition,weights:np.ndarray):
        weights = numpy_to_tensor(weights,self._device,torch.float32)
        self._intrinsic_embedding_optimizer.zero_grad(set_to_none=True)
        self._intrinsic_rnd_optimizer.zero_grad(set_to_none=True)
        rnd_loss = self._get_rnd_loss(transitions)
        embedding_loss = self._get_embedding_loss(transitions)
        rnd_loss = torch.mean((rnd_loss) * weights.detach())
        rnd_loss.backward()
        embedding_loss = torch.mean((embedding_loss) * weights.detach())
        embedding_loss.backward()
        # for name, param in self._rnd_predictor_network.named_parameters():
        #     if param.grad is None:
        #         print(f"No grad for RND predictor: {name}")
        #     else:
        #         print(f"Grad norm for RND predictor {name}: {param.grad.norm().item()}")
    
        # for name, param in self._embedding_network.named_parameters():
        #     if param.grad is None:
        #         print(f"No grad for Embedding network: {name}")
        #     else:
        #         print(f"Grad norm for Embedding network {name}: {param.grad.norm().item()}")
        if self._clip_gradients:
            torch.nn.utils.clip_grad_norm_(self._rnd_predictor_network.parameters(), self._max_gradient_norm)
            torch.nn.utils.clip_grad_norm_(self._embedding_network.parameters(), self._max_gradient_norm)
        # print(f"Intrinsic optimizer state: {self._intrinsic_optimizer.state_dict()}")
        
        self._intrinsic_embedding_optimizer.step()
        self._intrinsic_rnd_optimizer.step()
        # print(f"RND loss diff: {self._rnd_loss_t - rnd_loss.detach().mean().cpu()}")
        # print(f"Embedding loss diff: {self._embedding_network_loss_t - embedding_loss.detach().mean().cpu()}")
        # print(f"RND loss: {rnd_loss.mean().item()}")
        # print(f"Embedding loss: {embedding_loss.mean().item()}")
        self._rnd_loss_t = rnd_loss.detach().mean().cpu().item()
        self._embedding_network_loss_t = embedding_loss.detach().mean().cpu().item()
    def _get_rnd_loss(self,transition:NGUTransition)->torch.Tensor:
        state_t = numpy_to_tensor(transition.state_t[-5:],self._device,torch.float32)
        state_t = torch.flatten(state_t, 0,1)
        normed_state_t = self._normalise_rnd_observation(state_t)
        predicted_state_t = self._rnd_predictor_network(normed_state_t)
        with torch.no_grad():
            target_state_t = self._rnd_target_network(normed_state_t)
        rnd_loss = torch.square(predicted_state_t - target_state_t).mean(dim=1)
        rnd_loss = rnd_loss.view(5,-1)
        loss = torch.sum(rnd_loss,dim=0)
        return loss
    def _get_embedding_loss(self,transition:NGUTransition)->torch.Tensor:
        state_t = numpy_to_tensor(transition.state_t[-6:],self._device,torch.float32)
        action = numpy_to_tensor(transition.action_t[-6:],self._device,torch.int64)
        state_t_minus_1 = state_t[0:-1,...]
        state_t = state_t[1:,...]
        action_t_minus_1 = action[:-1,...]
        state_t_minus_1 = torch.flatten(state_t_minus_1,0,1)
        state_t = torch.flatten(state_t,0,1)
        action_t_minus_1 = torch.flatten(action_t_minus_1,0,1)
        
        embedding_state_t_minus_1 = self._embedding_network(state_t_minus_1)
        embedding_state_t = self._embedding_network(state_t)
        embeddings = torch.cat([embedding_state_t_minus_1,embedding_state_t],dim=-1)
        pi_logits = self._embedding_network.predict_action_logits(embeddings)
        loss = torch.nn.functional.cross_entropy(pi_logits,action_t_minus_1,reduction='none')
        loss = loss.view(5,-1)
        loss = torch.sum(loss,dim=0)
        return loss
    @torch.no_grad()
    def _normalise_rnd_observation(self,state_t:torch.Tensor)->torch.Tensor:
        rnd_observation = state_t.to(device=self._device,dtype=torch.float32)
        normed_observation = self._rnd_observation_normalizer.normalize(rnd_observation)
        normed_observation = normed_observation.clamp(-5,5)
        self._rnd_observation_normalizer.update(rnd_observation)
        return normed_observation
    @torch.no_grad()
    def _burn_in_unrolled_q_network(self,transitions:NGUTransition,initial_hidden_state:HiddenState)->Tuple[HiddenState,HiddenState]:
        state_t = numpy_to_tensor(transitions.state_t,self._device,torch.float32)
        last_action =numpy_to_tensor(transitions.last_action,self._device,torch.int64)
        extrinsic_reward = numpy_to_tensor(transitions.extrinsic_reward_t,self._device,torch.float32)
        intrinsic_reward = numpy_to_tensor(transitions.intrinsic_reward_t,self._device,torch.float32)
        policy_index = numpy_to_tensor(transitions.policy_index,self._device,torch.int64)
        
        _hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
        _target_hidden_state = tuple(s.clone().to(device=self._device) for s in initial_hidden_state)
        
        hidden_state = self._network(
            NGUNetworkInputs(
                state_t = state_t,
                action_t_minus_1 = last_action,
                extrinsic_reward_t = extrinsic_reward,
                intrinsic_reward_t = intrinsic_reward,
                policy_index_t = policy_index,
                hidden_state = _hidden_state,
            )
        ).hidden_state
        target_hidden_state = self._target_network(
            NGUNetworkInputs(
                state_t = state_t,
                action_t_minus_1 = last_action,
                extrinsic_reward_t = extrinsic_reward,
                intrinsic_reward_t = intrinsic_reward,
                policy_index_t = policy_index,
                hidden_state = _target_hidden_state,
            )
        ).hidden_state
        return hidden_state,target_hidden_state
    def _extract_hidden_state(self,transitions:NGUTransition)->np.ndarray:
        init_h = torch.from_numpy(transitions.init_h[0:1]).squeeze(0).to(device=self._device,dtype=torch.float32)
        init_c = torch.from_numpy(transitions.init_c[0:1]).squeeze(0).to(device=self._device,dtype=torch.float32)
        init_h = init_h.swapaxes(0,1)
        init_c = init_c.swapaxes(0,1)
        return init_h,init_c
    def _update_target_network(self):
        self._target_network.load_state_dict(self._network.state_dict())
        self._target_update_t +=1 
    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        return {
            'retrace_loss': self._retrace_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'embed_loss': self._embedding_network_loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }