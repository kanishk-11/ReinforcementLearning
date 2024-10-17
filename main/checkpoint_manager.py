
import os
from pathlib import Path
from typing import Any, Mapping, Text, Tuple

import torch
from main.common import AttributeDict

class PytorchCheckpointManager:
    def __init__(self,
                 environment_name:str, 
                 agent_name:str, 
                 save_dir:str,
                 iteration:int =0,
                 file_extension:str='ckpt',
                 restore_only:bool =False):
        self._save_dir = save_dir
        self.file_extension = file_extension
        self.base_path = None        
        if not restore_only and self._save_dir is not None and self._save_dir != '':
            self.base_path = Path(self._save_dir)
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

        self.state = AttributeDict()
        self.state.iteration = iteration
        self.state.environment_name = environment_name
        self.state.agent_name = agent_name
    def register_pair(self,pair:Tuple[Text,Any]):
        key, item = pair
        self.state[key] = item
    def save(self):
        if self.base_path is None:
            return
        file_name = f'{self.state.agent_name}_{self.state.environment_name}_{self.state.iteration}.{self.file_extension}'
        save_path = self.base_path/file_name
        states = self._get_states_dict()
        torch.save(states,save_path)
        return save_path
    def restore(self,file_to_restore):
        if not file_to_restore or not os.path.isfile(file_to_restore) or not os.path.exists(file_to_restore) :
            raise ValueError("Could not restore as file is not a valid checkpoint file")
        loaded_state = torch.load(file_to_restore,map_location=torch.device('cpu'))
        if loaded_state["environment_name"] != self.state.environment_name:
            raise RuntimeError(f'the checkpoint file does not belong to the same environment, checkpoint belongs to :{loaded_state["environment_name"]}')
        if 'agent_name' in loaded_state and loaded_state["agent_name"] != self.state.agent_name:
            raise RuntimeError(f'the checkpoint file does not belong to the same environment, checkpoint belongs to :{loaded_state["agent_name"]}')
        for key, value in loaded_state:
            if key not in self.state:
                continue
            else:
                if self._is_torch_nn(value):
                    self.state[key].load_state_dict(value)
                else:
                    self.state[key] = value
    
    def set_iteration(self,interation):
        self.state.iteration = interation
        
    def get_iteration(self):
        return self.state.iteration
            
    def _get_states_dict(self)-> Mapping[Text,Any]:
        states_dict = {}
        for key, item in self.state.items():
            if self._is_torch_nn(item):
                states_dict[key] = item.state_dict()
            else:
                states_dict[key] = item
        return states_dict
                
    def _is_torch_nn(self, item)->bool:
        return isinstance(item, (torch.nn.Module,torch.optim.Optimizer))
        