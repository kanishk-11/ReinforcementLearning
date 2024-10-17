from argparse import Action
from typing import Mapping, Text
import numpy as np
import torch
from main.abstractClasses.agent import Agent
from main.abstractClasses.timestep import TimeStepPair
from main.curiosity.episodic_bonus import EpisodicBonusModule
from main.curiosity.lifelong_bonus import RNDLifeLongBonusModule
from main.distributions import categorical_distribution
from main.networks.value_networks import NGUNetworkInputs
from models.common import numpy_to_tensor

def apply_egreedy_policy(
    q_values: torch.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> Action:
    """Apply e-greedy policy."""
    action_dim = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = q_values.argmax(-1).cpu().item()
    return a_t


class EpsilonGreedyActor(Agent):
    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 exploration_epsilon: float,
                 random_state:np.random.RandomState,
                 name:str):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=self._device)
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state
    
    def step(self,timestep:TimeStepPair) -> Action:
        return self._select_action(timestep)
        
    @torch.no_grad()
    def _select_action(self,timestep:TimeStepPair) -> Action:
        state_t = torch.tensor(timestep.observation[None,...]).to(device=self._device,dtype=torch.float32)
        q_values = self._network(state_t).q_values
        return apply_egreedy_policy(q_values,self._exploration_epsilon,self._random_state)
    def reset(self) -> None:
        """reset"""
        return
    def statistics(self) -> Mapping[str, float]:
        return {}

class DRQNEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 exploration_epsilon: float,
                 random_state:np.random.RandomState,
                 name: str):
        super().__init__(network,device,exploration_epsilon,random_state,name)
        self._lstm_state=None
    @torch.no_grad()
    def _select_action(self,timestep:TimeStepPair) -> Action:
        if self._lstm_state is None:
            raise ValueError("Reset Agent")
        state_t = torch.as_tensor(timestep.observation[None,None,...]).to(device=self._device,dtype=torch.float32)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        network_output = self._network(state_t,hidden_state)
        q_values = network_output.q_values
        self._lstm_state = network_output.hidden_state
        return apply_egreedy_policy(q_values,self._exploration_epsilon,self._random_state)
    def reset(self):
        self._lstm_state = self._network.get_initial_hidden_state(1)
        
class R2D2EpsilonGreedyActor(EpsilonGreedyActor):

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,
        device: torch.device,
    ):
        super().__init__(
            network=network,
            exploration_epsilon=exploration_epsilon,
            random_state=random_state,
            device=device,
            name='R2D2-greedy',
        )
        self._last_action = None
        self._lstm_state = None

    @torch.no_grad()
    def _select_action(self, timestep: TimeStepPair) -> Action:
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(self._last_action).to(device=self._device, dtype=torch.int64)
        r_t = torch.tensor(timestep.reward).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)

        network_output = self._network(
                state_t=s_t[None, ...],
                action_t_minus_1=a_tm1[None, ...],
                reward_t=r_t[None, ...],
                hidden_state=hidden_s,
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_state

        a_t = apply_egreedy_policy(q_t, self._exploration_epsilon, self._random_state)
        self._last_action = a_t
        return a_t

    def reset(self) -> None:
        self._last_action = 0 
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
class NGUEpsilonGreedyActor(EpsilonGreedyActor):
    def __init__(
        self,
        network:torch.nn.Module,
        embedding_network:torch.nn.Module,
        rnd_target_network:torch.nn.Module,
        rnd_predictor_network:torch.nn.Module,
        episodic_memory_size:int,
        num_neighbors:int,
        cluster_distance:float,
        kernel_epsilon:float,
        maximum_similarity:float,
        exploration_epsilon:float,
        random_state:np.random.RandomState,
        device:torch.device
    ):
        super().__init__(
            network=network,
            exploration_epsilon=exploration_epsilon,
            random_state=random_state,
            device=device,
            name='NGU-greedy',
        )
        self._policy_index = 0
        self._policy_beta = 0
        self._episodic_module = EpisodicBonusModule(
            embedding_network=embedding_network,
            device=device,
            size=episodic_memory_size,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            maximum_similarity=maximum_similarity
        )
        self._lifelong_module = RNDLifeLongBonusModule(
            target_network=rnd_target_network,
            predictor_network=rnd_predictor_network,
            device=device,
            discount=0.99
        )
        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
    @torch.no_grad()
    def step(self,timestep:TimeStepPair) -> Action:
        action_t = self._select_action(timestep=timestep)
        state_t = numpy_to_tensor(array = timestep.observation[None,...],device=self._device,dtype=torch.float32)
        self._lifelong_bonus_t = self._lifelong_module.get_bonus(state_t)
        self._episodic_bonus_t = self._episodic_module.get_bonus(state_t)
        return action_t
    @torch.no_grad()
    def _select_action(self,timestep:TimeStepPair) -> Action:
        state_t = torch.tensor(timestep.observation[None,...]).to(device=self._device,dtype=torch.float32)
        action_t_minus_1 = torch.tensor(self._last_action).to(device=self._device,dtype=torch.int64)
        extrinsic_reward = torch.tensor(timestep.reward).to(device=self._device,dtype=torch.float32)
        intrinsic_reward = torch.tensor(self.intrinsic_reward).to(self._device,torch.float32)
        policy_index = torch.tensor(self._policy_index).to(device=self._device,dtype=torch.int64)
        hidden_state = tuple(s.to(device=self._device) for s in self._lstm_state)
        pi_output = self._network(
            NGUNetworkInputs(
                state_t=state_t[None,...],
                action_t_minus_1=action_t_minus_1[None,...],
                extrinsic_reward_t=extrinsic_reward[None,...],
                intrinsic_reward_t=intrinsic_reward[None,...],
                policy_index_t=policy_index[None,...],
                hidden_state=hidden_state,
            )
        )
        q_t = pi_output.q_values
        self._lstm_state = pi_output.hidden_state
        action_t = apply_egreedy_policy(q_t,epsilon=self._exploration_epsilon,random_state=self._random_state)
        self._last_action = action_t
        return action_t
    def reset(self):
        self._last_action = 0
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)
        self._episodic_module.reset()
    @property
    def intrinsic_reward(self) -> torch.Tensor:
        return self._episodic_bonus_t * min(max(self._lifelong_bonus_t,1.0),5.0)
    
class PolicyGreedyActor(Agent):
    """Agent that acts with a given set of policy network parameters."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device,
        name: str = '',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)

    def step(self, timestep:TimeStepPair) -> Action:
        """Give current timestep, return best action"""
        return self.act(timestep)

    def act(self, timestep: TimeStepPair) -> Action:
        """Selects action given a timestep."""
        return self._select_action(timestep)

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """

    @torch.no_grad()
    def _select_action(self, timestep: TimeStepPair) -> Action:
        """Samples action from policy at given state."""
        s_t = torch.tensor(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        pi_logits_t = self._network(s_t).pi_logits

        # Sample an action
        a_t = categorical_distribution(pi_logits_t).sample()

        # # Can also try to act greedy
        # prob_t = F.softmax(pi_logits, dim=1)
        # a_t = torch.argmax(prob_t, dim=1)

        return a_t.cpu().item()

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Empty statistics"""
        return {}