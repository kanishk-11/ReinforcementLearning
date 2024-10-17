
from asyncio.log import logger
import collections
from typing import Iterable
from main.ReplayUtils.common import get_n_step_transition
from main.ReplayUtils.transition import Transition
from main.abstractClasses.timestep import TimeStepPair


class TransitionAccumulator():
    def __init__(self):
        self.timestep_t_minus_1:TimeStepPair = None
        self.action_t_minus_1:int = None
    def reset(self):
        self.__init__()
    def step(self,timestep_t:TimeStepPair,action_t:int)->Iterable[Transition]:
        if timestep_t.first:
            self.reset()
        
        if self.timestep_t_minus_1 is None:
            if not timestep_t.first:
                raise ValueError("Transitions must be accumulated sequentially")
            self.timestep_t_minus_1 = timestep_t
            self.action_t_minus_1 = action_t
            return
        # logger.info(len(self.timestep_t_minus_1.observation))
        transition = Transition(
            state_t_minus_1= self.timestep_t_minus_1.observation,
            action_t_minus_1=self.action_t_minus_1,
            reward_t=timestep_t.reward,
            state_t=timestep_t.observation,
            done= timestep_t.done
        )
        self.timestep_t_minus_1 = timestep_t
        self.action_t_minus_1 = action_t
        yield transition
    
class NStepTransitionAccumulator():
    def __init__(self,n,discount) -> None:
        self.discount = discount
        self.transitions = collections.deque(maxlen=n)
        self.timestep_t_minus_1 = None
        self.action_t_minus_1 = None
    def step(self,timestep_t:TimeStepPair,action_t:int)->Iterable[Transition]:
        if timestep_t.first:
            self.reset()

        if self.timestep_t_minus_1 is None:
            if not timestep_t.first:
                raise ValueError("Transitions must be accumulated sequentially")
            self.action_t_minus_1= action_t
            self.timestep_t_minus_1 = timestep_t
            return
        self.transitions.append(
            Transition(
                state_t_minus_1=self.timestep_t_minus_1.observation,
                action_t_minus_1=self.action_t_minus_1,
                reward_t=timestep_t.reward,
                state_t=timestep_t.observation,
                done=timestep_t.done
            )
        )
        self.action_t_minus_1  =action_t
        self.timestep_t_minus_1 = timestep_t

        if timestep_t.done:
            while self.transitions:
                yield get_n_step_transition(self.transitions,self.discount)
                self.transitions.popleft()
        else:
            if len(self.transitions) < self.transitions.maxlen:
                return
            assert self.transitions.maxlen == len(self.transitions)
            yield get_n_step_transition(self.transitions,self.discount)
    def reset(self):
        self.transitions.clear()
        self.timestep_t_minus_1 = None
        self.action_t_minus_1 = None



    
        

    