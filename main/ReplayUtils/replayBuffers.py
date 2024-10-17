
from enum import Enum
from typing import Any, Callable, Generic, List, Mapping, Optional, Sequence
from venv import logger

import numpy as np
from torch import fill
from main.ReplayUtils.common import ReplayData, stack_transitions
from main.abstractClasses.replayBuffer import ReplayBuffer

class UniformReplay(Generic[ReplayData],ReplayBuffer):
    def __init__(
            self,
            size:int,
            structure:ReplayData,
            random_state: np.random.RandomState,   #For reproducing results.
            time_major: bool = False,              #For RNNs.
            encoder : Optional[Callable[[ReplayData],Any]] = None,
            decoder : Optional[Callable[[Any],ReplayData]] = None,
            device_mode : str="np"
    ):
        if size <=0:
            raise ValueError("Replay buffer size must be > 0 , size parameter passed: {}".format(size))
        self.structure = structure
        self.size = size
        self._random_state = random_state
        self._transitions = [None]*size
        self.transitions_added = 0
        self._time_major = time_major
        self._encoder = encoder or (lambda x:x)
        self._decoder = decoder or (lambda x:x)
        self._device_mode = device_mode
    
    def add(self, data: ReplayData):
        self._transitions[self.transitions_added%self.size] = self._encoder(data)
        self.transitions_added += 1
        
    def _get(self, index: Sequence[int])->List[ReplayData]:
        return [self._decoder(self._transitions[index[i]]) for i in range(len(index))]
    
    def sample(self, batch_size: int)->List[ReplayData]:
        if batch_size <=0:
            raise ValueError("Batch size must be > 0 , batch_size parameter passed: {}".format(batch_size))
        if batch_size > self.transitions_added:
            raise ValueError("Batch size cannot be greater than replay buffer size or number of transitions added")
        indices = self._random_state.randint(0, min(self.transitions_added,self.size), size=batch_size)
        samples = self._get(indices)
        return stack_transitions(samples, self.structure, self.stacking_dimension,mode=self._device_mode)
    
    
    @property
    def stacking_dimension(self) -> int:
        if self._time_major:
            return 1
        else:
            return 0
    @property
    def filled_size(self) -> int:
        return min(self._states_added, self.size)

PriorizationMode = Enum('PrioritizationMode',['PROPORTIONAL','RANK'])  
  
class PrioritizedReplay():
    def __init__(
            self,
            size:int,
            structure:ReplayData,
            priority_exponent: float,
            importance_sampling_exponent:float,
            random_state: np.random.RandomState,
            normalize_weights: bool = True,
            time_major : bool = False,
            encoder : Optional[Callable[[ReplayData],Any]] = None,
            decoder : Optional[Callable[[Any],ReplayData]] = None,
            config : Mapping[str,Any] = {
                'mode':"np",
                'priorirization_mode':PriorizationMode.PROPORTIONAL
            }
    ):
        if size <=0:
            raise ValueError("Replay buufer size must be > 0 , size parameter passed: {}".format(size))
        self.structure = structure
        self.size = size
        self._random_state = random_state
        self._transitions = [None]*size
        self.transitions_added = 0
        self._time_major = time_major
        self._encoder = encoder or (lambda x:x)
        self._decoder = decoder or (lambda x:x)
        
        self._validateConfigKeys(config=config)
        if config.get('mode') in ["np","torch"]:
            self._device_mode = config['mode']
        else:
            raise ValueError("Invalid prioritization mode, mode parameter passed: {}".format(self.priortization_mode))
        
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent
        self.normalize_weights = normalize_weights
        self._priorities = np.ones((size,), dtype=np.float32)
        if config.get('priorirization_mode') in [PriorizationMode.PROPORTIONAL,PriorizationMode.RANK]:
            self.priortization_mode = config.get('priorirization_mode')
        else:
            raise ValueError("Invalid prioritization mode, mode parameter passed: {}".format(self.priortization_mode))

    def add(self, data: ReplayData, priority: float):
        if not np.isfinite(priority):
            raise ValueError("Priority must be finite, priority parameter passed: {}".format(priority))
        if priority<0.0:
            raise ValueError("Priority must be positive, priority parameter passed: {}".format(priority))
                
        self._transitions[self.transitions_added%self.size] = self._encoder(data)
        self._priorities[self.transitions_added%self.size] = priority
        self.transitions_added += 1 
    
    def get(self, index: Sequence[int])->List[ReplayData]:
        return [self._decoder(self._transitions[index[i]]) for i in range(len(index))]
    
    def sample(self, batch_size: int):
        """Samples a batch of transitions"""
        if batch_size <=0:
            raise ValueError("Batch size must be > 0 , batch_size parameter passed: {}".format(batch_size))
        if batch_size > self.transitions_added:
            raise ValueError("Batch size cannot be greater than replay buffer size or number of transitions added")
        filled_indices = min(self.size,self.transitions_added)
        if self._priority_exponent==0:
            """uniform sampling, avoid computation"""
            indices = self._random_state.randint(0, filled_indices, size=batch_size).astype(np.int64)
            weights = np.ones_like(indices, dtype=np.float32)
        else:
            if self.priortization_mode == PriorizationMode.PROPORTIONAL:
                priorities = self._priorities[:filled_indices]**self._priority_exponent
                probabilities = priorities/np.sum(priorities)
            else:
                priorities = [(priority, i) for i, priority in enumerate(self._priorities[:filled_indices])]
                priorities = np.sort(priorities)
                actual_to_current_index_map = {}
                for i, (priority, index) in enumerate(priorities):
                    actual_to_current_index_map[index] = i
                probabilities = [1.0/ actual_to_current_index_map.get(i,0.0) for i in range(filled_indices)]
                del actual_to_current_index_map

            indices = self._random_state.choice(np.arange(probabilities.shape[0]),size=batch_size, p=probabilities, replace=True)
            weights = ((1.0 / self.size) / np.take(probabilities,indices))**self.importance_sampling_exponent
            if self.normalize_weights:
                weights /= np.max(weights)
        transitions = self.get(indices)
        stacked_transitions = stack_transitions(transitions=transitions,structure=self.structure,axis=self.stacking_dimension,mode=self._device_mode)
        return stacked_transitions,indices,weights
    
    def update_priorities(self,indices:Sequence[int],priorities:Sequence[float]):
        priorities = np.asarray(priorities)
        if not np.isfinite(priorities).all() or (priorities < 0).any():
            raise ValueError("Priorities must be finite and positive")
        for index, priority in zip(indices,priorities):
            self._priorities[index] = priority
    def _validateConfigKeys(self, config):
        logger.info(f'init buffer with config: {config}')
        required_keys = ['mode', 'priorirization_mode']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Required key '{key}' not specified in config.")
    @property
    def importance_sampling_exponent(self) -> float:
        return self._importance_sampling_exponent(self.transitions_added)
    @property
    def stacking_dimension(self) -> int:
        if self._time_major:
            return 1
        else:
            return 0
    
    

