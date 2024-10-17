
from typing import NamedTuple, Optional

import numpy as np


class Transition(NamedTuple):
    """Represents a single transition in the replay buffer."""
    state_t_minus_1: Optional[np.ndarray]
    action_t_minus_1 : Optional[int]
    reward_t : Optional[float]
    state_t : Optional[np.ndarray]
    done : Optional[bool]

