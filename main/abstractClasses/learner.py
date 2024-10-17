import abc
from typing import Any, Iterable, Mapping, Text


class Learner(abc.ABC):
    """Learner interface."""

    agent_name: str  # agent name
    step_t: int  # learner steps

    @abc.abstractmethod
    def step(self) -> Iterable[Mapping[Text, float]]:
        """Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred, otherwise None.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Should be called at the beginning of every iteration."""

    @abc.abstractmethod
    def received_item_from_queue(self, item: Any) -> None:
        """Received item send by actors through multiprocessing queue."""

    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""