from typing import Any, Mapping, NamedTuple, Optional, Text
import numpy as np
class TimeStepPair(NamedTuple):
    """Enviroment timestep details"""
    observation: Optional[np.ndarray]
    reward: Optional[float]
    done : Optional[bool]
    first: Optional[bool]
    info : Optional[Mapping[Text,Any]]
