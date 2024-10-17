
import collections
import copy
import itertools
from typing import Any
from main.ReplayUtils.common import ReplayData, stack_transitions


class Unroller:
    def __init__(self,unroll_length:int, overlap:int,structure:ReplayData,cross_episode:bool):
        self.unroll_length = unroll_length
        self._full_unroll_length = overlap +unroll_length
        self.overlap = overlap
        self.structure = structure
        self.cross_episode = cross_episode
        self._internal_queue = collections.deque(maxlen=self._full_unroll_length)
        self._last_unroll = None
    
    @property
    def full(self) -> bool:
        return len(self._internal_queue) == self._internal_queue.maxlen
    @property
    def size(self)-> int:
        return len(self._internal_queue)
    def add(self,transition:Any, done:bool):
        self._internal_queue.append(transition)
        if self.full:
            return self._pack_unroll()
        if done:
            return self._episode_end()
        return None

    def _pack_unroll(self):
        if not self.full:
            return None
        sequence = list(self._internal_queue)
        self._last_unroll = copy.deepcopy(sequence)
        self._internal_queue.clear()
        if self.overlap > 0:
            for transtion in sequence[-self.overlap:]:
                self._internal_queue.append(transtion)
        return self._stack_unroll(sequence)
    def _episode_end(self):
        if self.cross_episode:
            return None
        if self.size >0 and self._last_unroll is not None:
            suffix = list(self._internal_queue)
            prefix_len = self._full_unroll_length - len(suffix)
            prefix = self._last_unroll[-prefix_len:]
            return_sequence = list(itertools.chain(prefix,suffix))
            return self._stack_unroll(return_sequence)
        else:
            return None
    def _stack_unroll(self,sequence:list)-> Any:
        return stack_transitions(transitions=sequence,structure=self.structure)
    def reset(self):
        self._last_unroll = None
        self._internal_queue.clear()
    