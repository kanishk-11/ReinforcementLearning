import abc
from typing import Mapping,Text

from main.abstractClasses.action import Action
from main.abstractClasses.timestep import TimeStepPair

class Agent(abc.ABC):
    agent_name: str
    step_t : int
    @abc.abstractmethod
    def step(self, timestep_pair:TimeStepPair)->Action:
        """Implements a step function for the agent."""
    @abc.abstractmethod
    def reset(self)->None:
        """Resets the agent."""
        raise NotImplementedError("step() function not implemented in the base Agent class for" + self.agent_name)
    @abc.abstractmethod
    def statistics(self)->Mapping[Text,float]:
        """Returns a dictionary of statistics about the agent."""
        raise NotImplementedError("step() function not implemented in the base Agent class for" + self.agent_name)



        