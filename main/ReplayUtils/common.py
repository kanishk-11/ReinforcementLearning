
import logging
from typing import Any, Iterable, Tuple, TypeVar

import numpy as np
import snappy
import torch

from main.ReplayUtils.transition import Transition


CompressedArray =  Tuple[bytes, Tuple, np.dtype]
ReplayData = TypeVar('ReplayData',bound=Tuple[Any,...])

def stack_transitions(transitions,structure,axis=0,mode = "np"):
    # logging.info(len(transitions))
    transposed = zip(*transitions)
    if(mode == "np"):        
        stacked = [np.stack(xs, axis=axis) for xs in transposed]
    elif mode == "torch":
        stacked = [torch.stack(xs, dim=axis) for xs in transposed]
    else:
        raise ValueError(f"Unsupported mode {mode}")
    # logging.info(len(stacked))
    # logging.info(type(structure))
    return type(structure)(*stacked)

def get_n_step_transition(transitions:Iterable[Transition],discount_factor:float)->Transition:
    """Build 1 n step transition for n 1 step transitions"""
    reward_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        reward_t += transition.reward_t*discount_t
        discount_t*=discount_factor
    return Transition(state_t_minus_1=transitions[0].state_t_minus_1,
                      action_t_minus_1=transition[0].action_t_minus_1,
                      reward_t=reward_t,
                      state_t=transitions[-1].state_t,
                      done=transitions[-1].done)
    
def compress(array: np.ndarray) -> CompressedArray:
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def decompress(compressed: CompressedArray) -> np.ndarray:
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)

def split_from_structure(input_, structure, prefix_length: int, axis: int = 0) -> Tuple[ReplayData]:
    if prefix_length > 0:
        for v in input_:
            if v.shape[axis] < prefix_length:
                raise ValueError(f'Expected prefix length to be less than {v.shape[axis]}')
    if prefix_length == 0:
        return (None,input_)
    else:
        split = [np.split(
            xs,
            [prefix_length,xs.shape[axis]],
            axis=axis
        )
        for xs in input_]
        _prefix = [pair[0] for pair in split]
        _suffix = [pair[1] for pair in split]
        return (type(structure)(*_prefix), type(structure)(*_suffix))
        