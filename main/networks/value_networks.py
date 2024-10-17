
from asyncio.log import logger
from typing import NamedTuple, Optional, Tuple

import torch

from main.networks.common import BackboneConv, init_weights_of_net

class RNNDQNOutputs(NamedTuple):
    q_values: torch.Tensor
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
class DQNOutput(NamedTuple):
    q_values: torch.Tensor
class NGUNetworkInputs(NamedTuple):
    state_t: torch.Tensor
    action_t_minus_1: torch.Tensor
    extrinsic_reward_t: torch.Tensor
    intrinsic_reward_t: torch.Tensor
    policy_index_t: torch.Tensor
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
class DQNNet(torch.nn.Module):
    def __init__(self,state_dimension:int,action_dimension:int):
        if action_dimension <=0:
            raise ValueError("action dimension must be greater than 0")
        if state_dimension <= 0:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dimension)
        )
    
    def forward(self,state:torch.Tensor)->DQNOutput:
        q_values = self.network(state)
        return DQNOutput(q_values)
class DQNNetAA(torch.nn.Module):
    def __init__(self,state_dimension:int,action_dimension:int,mode="MAX"):
        if action_dimension <=0:
            raise ValueError("action dimension must be greater than 0")
        if state_dimension <= 0:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.mode = mode
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1)
        )
        self.advantage_network = torch.nn.Sequential(
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,action_dimension)
        )
    
    def forward(self,state:torch.Tensor)->DQNOutput:
        values = self.network(state)
        state_values = self.value_network(values)
        advantages = self.advantage_network(values)
        if self.mode =="MAX":
            max_values,_ = torch.max(advantages,dim=-1,keepdim=True)
            q_values = state_values + advantages - max_values
        else:
            q_values = state_values + advantages - torch.mean(advantages,dim=-1,keepdim=True)
        return DQNOutput(q_values)
    
class DQNConvNet(torch.nn.Module):
    def __init__(self,state_dim:tuple,action_dimension:int):
        if action_dimension <=0:
            raise ValueError("action dimension must be greater than 0")
        if len(state_dim) != 3:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.action_dim = action_dimension
        self.main_body = BackboneConv(state_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.main_body.out_features,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,self.action_dim)
        )
        init_weights_of_net(self)
    
    def forward(self,state:torch.Tensor):
        # logger.info(f'self.main_body.out_features:{self.main_body.out_features})')
        # logger.info(f'state.shape:{state.shape}')
        state = state.float()/255
        state = self.main_body(state)
        # logger.info(f'state.shape[after CNN]:{state.shape}')
        q_values = self.head(state)
        return DQNOutput(q_values)
        
class DQNConvNetAA(torch.nn.Module):
    def __init__(self,state_dim:tuple,action_dimension:int,mode="MAX"):
        if action_dimension <=0:
            raise ValueError("action dimension must be greater than 0")
        if len(state_dim) != 3:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.action_dim = action_dimension
        self.main_body = BackboneConv(state_dim)
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(self.main_body.out_features,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024,self.action_dim)
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.main_body.out_features,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1)
        )
        self.mode = mode
        init_weights_of_net(self)
    
    def forward(self,state:torch.Tensor):
        # logger.info(f'self.main_body.out_features:{self.main_body.out_features})')
        # logger.info(f'state.shape:{state.shape}')
        state = state.float()/255
        state = self.main_body(state)
        # logger.info(f'state.shape[after CNN]:{state.shape}')
        state_value = self.value_head(state)
        state_advantages = self.advantage_head(state)
        if self.mode =="MAX":
            max_values,_ = torch.max(state_advantages,dim=-1,keepdim=True)
            value = state_value + state_advantages - max_values
        else:
            value = state_value + state_advantages - torch.mean(state_advantages,dim=-1,keepdim=True)
        return DQNOutput(value)
class DRQNet(torch.nn.Module):
    def __init__(self,state_dim :int, action_dim :int):
        if action_dim < 1:
            raise ValueError("action dimension must be greater than 0")
        if state_dim < 1:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.action_dim = action_dim
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(input_size=64,hidden_size=64, num_layers=1, batch_first=True )
        self.value_head = torch.nn.Linear(self.lstm.hidden_size,action_dim)
    
    def forward(self,x:torch.Tensor,hidden:torch.Tensor):
        B,T = x.shape[:2]
        x = torch.flatten(x,0,1)
        x = self.feature_extractor(x)
        x = x.view(B,T,-1)
        x,hidden_state = self.lstm(x,hidden)
        x= torch.flatten(x,0,1)
        q_values = self.value_head(x)
        q_values = q_values.view(B,T,-1)
        return RNNDQNOutputs(q_values, hidden_state)
        
    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
    
class DRQNConv(torch.nn.Module):
    def __init__(self,state_dim :int, action_dim :int):
        if action_dim < 1:
            raise ValueError("action dimension must be greater than 0")
        if len(state_dim) != 3:
            raise ValueError("state dimension len must be 3")
        super().__init__()
        self.action_dim = action_dim
        self.feature_extractor = BackboneConv(state_dim)
        self.lstm = torch.nn.LSTM(self.feature_extractor.out_features,hidden_size=256,num_layers=1,batch_first=True)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,action_dim)
        )
        init_weights_of_net(self)
    
    def forward(self,x:torch.Tensor,hidden_state=None):
        B,T = x.shape[:2]
        x = torch.flatten(x,0,1)
        x = x.float()/255
        x = self.feature_extractor(x)
        x = x.view(B,T,-1)
        x, hidden_state = self.lstm(x,hidden_state)
        x = torch.flatten(x,0,1)
        q_values = self.value_head(x)
        q_values = q_values.view(B,T,-1)
        return RNNDQNOutputs(q_values=q_values,hidden_state=hidden_state)
    
    def get_initial_hidden_state(self,batch_size:int):
        return tuple(torch.zeros(self.lstm.num_layers,batch_size,self.lstm.hidden_size) for _ in range(2))
    
class R2D2DQN(torch.nn.Module):
    def __init__(self,state_dim :int, action_dim :int):
        if action_dim<1:
            raise ValueError("action dimension must be greater than 0")
        if state_dim<1:
            raise ValueError("state dimension must be greater than 0")
        super().__init__()
        self.action_dim = action_dim
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(state_dim,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(128 + action_dim + 1 ,hidden_size=128, num_layers=1)
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(128,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1)
        )
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,action_dim)
        )
    def forward(self, state_t, action_t_minus_1,reward_t,hidden_state):
        T,B,*_ = state_t.shape
        x = torch.flatten(state_t,0,1)
        x = self.feature_extractor(x)
        x = x.view(T*B,-1)
        action_t_minus_1_one_hot_encoded = torch.nn.functional.one_hot(action_t_minus_1.view(T*B),self.action_dim).float().to(device=x.device)
        reward = reward_t.view(T*B,1)
        core_input = torch.cat([x,action_t_minus_1_one_hot_encoded,reward],dim=-1)
        core_input = core_input.view(T,B,-1)
        if hidden_state is None:
            hidden_state = self.get_initial_hidden_state(batch_size=B)
            hidden_state = tuple(s.to(device=x.device) for s in hidden_state)

        x, hidden_state = self.lstm(core_input, hidden_state)
        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]
        q_values = values + (advantages - torch.mean(advantages,dim=-1,keepdim=True))
        q_values = q_values.view(T, B, -1)  # reshape to in the range [B, T, action_dim]
        return RNNDQNOutputs(q_values=q_values, hidden_state=hidden_state)
    
    def get_initial_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))                

class R2D2DQNConv(torch.nn.Module):
    def __init__(self,state_dim :int, action_dim :int):
        if action_dim<1:
            raise ValueError("action dimension must be greater than 0")
        if len(state_dim) != 3:
            raise ValueError("state dimension len must be 3")
        super().__init__()
        self.action_dim = action_dim
        self.feature_extractor = BackboneConv(state_dim)
        self.lstm = torch.nn.LSTM(self.feature_extractor.out_features + action_dim + 1,hidden_size=512,num_layers=1)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,1)
        )
        self.advantage_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,action_dim)
        )
        init_weights_of_net(self)
    
    def forward(self, state_t:torch.Tensor, action_t_minus_1:torch.Tensor,reward_t:torch.Tensor,hidden_state:Optional[Tuple[torch.Tensor, torch.Tensor]]):
        T,B,*_ = state_t.shape
        x = torch.flatten(state_t,0,1)
        x = x.float()/255.0
        x = self.feature_extractor(x)
        x = x.view(T*B,-1)
        action_t_minus_1_one_hot_encoded = torch.nn.functional.one_hot(action_t_minus_1.view(T*B),self.action_dim).float().to(device=x.device)
        reward = reward_t.view(T*B,1)
        core_input = torch.cat([x,reward,action_t_minus_1_one_hot_encoded],dim=-1)
        core_input = core_input.view(T,B,-1)
        
        if hidden_state is None:
            hidden_state = self.get_initial_hidden_state(batch_size=B)
            hidden_state = tuple(s.to(device=x.device) for s in hidden_state)
        x, hidden_state = self.lstm(core_input, hidden_state)
        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]
        
        q_values = values + (advantages - torch.mean(advantages,dim=-1,keepdim=True))
        q_values = q_values.view(T, B, -1)  # reshape to in the range [B, T, action_dim]
        return RNNDQNOutputs(q_values=q_values, hidden_state=hidden_state)

    def get_initial_hidden_state(self,batch_size) -> Tuple[torch.Tensor,torch.Tensor]:
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))

class NGUConv(torch.nn.Module):
    def __init__(self,state_dim:int,action_dim:int,num_policies:int):
        super().__init__()
        self.action_dim = action_dim
        self.num_policies = num_policies
        self.feature_extractor = BackboneConv(state_dim)
        output_size = self.feature_extractor.out_features + self.num_policies + self.action_dim + 1 + 1
        self.lstm = torch.nn.LSTM(input_size=output_size,hidden_size=512,num_layers=1)
        self.advatage_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,action_dim)
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.lstm.hidden_size,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,1)
        )
        init_weights_of_net(self)
    def forward(self,input_:NGUNetworkInputs):
        state_t = input_.state_t
        action_t_minus_1 = input_.action_t_minus_1
        extrinsic_reward_t = input_.extrinsic_reward_t
        intrinsic_reward_t = input_.intrinsic_reward_t
        policy_index = input_.policy_index_t
        hidden_state = input_.hidden_state
        T,B,*_ = state_t.shape
        x = torch.flatten(state_t,0,1)
        x = x.float()/255.0
        x = self.feature_extractor(x)
        x = x.view(T*B,-1)
        one_hot_beta_encoding = torch.nn.functional.one_hot(policy_index.view(T*B),self.num_policies).float().to(device=x.device)
        one_hot_action_t_minus_1_encoding = torch.nn.functional.one_hot(action_t_minus_1.view(T*B),self.action_dim).float().to(x.device)
        intrinsic_reward = intrinsic_reward_t.view(T*B,1)
        extrinsic_reward = extrinsic_reward_t.view(T*B,1)
        input_to_net = torch.cat([x,one_hot_action_t_minus_1_encoding,one_hot_beta_encoding,intrinsic_reward,extrinsic_reward],dim=-1)
        input_to_net = input_to_net.view(T,B,-1)
        if hidden_state is None:
            hidden_state = self.get_initial_hidden_state(batch_size=B)
            hidden_state = tuple(s.to(device=x.device) for s in hidden_state)
        x,hidden_state = self.lstm(input_to_net,hidden_state)
        x = torch.flatten(x,0,1)
        advantages = self.advatage_head(x)
        values = self.value_head(x)

        q_values = values + (advantages - torch.mean(advantages,dim=-1,keepdim=True))
        q_values = q_values.view(T, B, -1)  # reshape to in the range [B, T, action_dim]
        return RNNDQNOutputs(q_values=q_values,hidden_state=hidden_state)
    def get_initial_hidden_state(self,batch_size):
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))
