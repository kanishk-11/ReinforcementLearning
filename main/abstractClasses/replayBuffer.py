import abc
from typing import Any, Callable, Optional
from main.ReplayUtils.common import ReplayData
import numpy as np
class ReplayBuffer(abc.ABC):
    replay_buffer_name : str
    size:int
    _random_state : np.random.RandomState
    time_major : bool
    encoder : Optional[Callable[[ReplayData], Any]]
    decoder : Optional[Callable[[Any],ReplayData]]
    @abc.abstractmethod
    def add(self, **args):
        """Add experience to the buffer"""
    @abc.abstractmethod
    def sample(self, batch_size:int):
        """Sample a batch of experiences"""
    