
from asyncio.log import logger
import collections
from pathlib import Path
import shutil
import timeit
from typing import Any, Iterable, Mapping, Optional, Text, Tuple, Union

import gymnasium
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from main.ReplayUtils.transition import Transition

class EpisodeTracker:
    def __init__(self) -> None:
        self._num_steps_after_reset = 0
        self._episode_returns = None
        self._episode_steps = None
        self._episode_visited_rooms = None
        self._current_episode_rewards = None
        self._current_episode_step = None

    def step(self,env,timestep_t, agent,action_t):
        del (env,agent,action_t)
        if timestep_t.first:
            if self._current_episode_rewards:
                raise ValueError('current episode reward list should be empty')
            if self._current_episode_step:
                raise ValueError('current episode step should be 0')
        else:
            reward = timestep_t.reward
            if isinstance(timestep_t.info,dict) and 'raw_reward' in timestep_t.info:
                reward = timestep_t.info['raw_reward']
            self._current_episode_rewards.append(reward)
        
        self._num_steps_after_reset +=1
        self._current_episode_step +=1
        if timestep_t.done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step=0
            #Some environments in atari have this.
            if isinstance(timestep_t.info, dict) and 'episode_visited_rooms' in timestep_t.info:
                self._episode_visited_rooms.append(timestep_t.info['episode_visited_rooms'])
    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._episode_visited_rooms = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
    
    def get(self) -> Mapping[str,Union[int, float, None]]:
        mean_episode_visited_rooms = 0
        if len(self._episode_returns) > 0:
            mean_episode_return = np.array(self._episode_returns).sum()
            if len(self._episode_visited_rooms) > 0:
                mean_episode_visited_rooms = np.array(self._episode_visited_rooms).mean()
        else:
            mean_episode_return = sum(self._current_episode_rewards)
        return {
            'mean_episode_return': mean_episode_return,
            'mean_episode_visited_rooms': mean_episode_visited_rooms,
            'num_episodes': len(self._episode_returns),
            'current_episode_step': self._current_episode_step,
            'num_steps_since_reset': self._num_steps_since_reset,
        }
    

class StepRateTracker:
    def __init__(self):
        self._num_steps_after_reset = None
        self._start = None

    def step(self, env, timestep_t, agent, a_t) -> None:
        """Accumulates statistics from timestep."""
        del (env, timestep_t, agent, a_t)

        self._num_steps_after_reset += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._num_steps_after_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_after_reset > 0:
            step_rate = self._num_steps_after_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps_after_reset': self._num_steps_after_reset,
            'duration': duration,
        } 
    
class TensorboardEpisodeTracker(EpisodeTracker):

    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self._total_steps = 0  
        self._total_episodes = 0 
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            self._total_episodes += 1
            tb_steps = self._total_steps

            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            self._writer.add_scalar('performance(env_steps)/num_episodes', self._total_episodes, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_return', episode_return, tb_steps)
            self._writer.add_scalar('performance(env_steps)/episode_steps', episode_step, tb_steps)

            if isinstance(timestep_t.info, dict) and 'episode_visited_rooms' in timestep_t.info:
                episode_visited_rooms = self._episode_visited_rooms[-1]
                self._writer.add_scalar('performance(env_steps)/episode_visited_rooms', episode_visited_rooms, tb_steps)

class TensorboardStepRateTracker(StepRateTracker):

    def __init__(self, writer: SummaryWriter):
        super().__init__()

        self._total_steps = 0  # keep track total number of steps, does not reset
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            time_stats = self.get()
            self._writer.add_scalar('performance(env_steps)/step_rate', time_stats['step_rate'], self._total_steps)

class TensorboardAgentStatisticsTracker:

    def __init__(self, writer: SummaryWriter):
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._writer = writer

    def step(self, env, timestep_t, agent, a_t) -> None:
        del (env, a_t)
        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        # This should not block the training loop if there's any exception.
        if timestep_t.done:
            try:
                stats = agent.statistics
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'agent_statistics(env_steps)/{k}', v, self._total_steps)
            except Exception:
                pass

    def reset(self) -> None:
        pass

    def get(self) -> Mapping[Text, float]:
        return {}
class TensorboardScreenshotTracker:

    def __init__(self, writer: SummaryWriter, log_interval: int = 100):
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._total_episodes = 0  # keep track total number of episodes, does not reset
        self._log_interval = log_interval
        self._writer = writer

    def step(self, env:gymnasium.Env, timestep_t, agent, a_t) -> None:
        del (agent, a_t)

        self._total_steps += 1

        if timestep_t.done:
            self._total_episodes += 1

            if self._total_episodes % self._log_interval == 0:
                try:
                    img = env.render()
                    # logger.info(img)
                    self._writer.add_image(
                        f'debug(episode)/episode_{self._total_episodes}',
                        img,
                        self._total_steps,
                        dataformats='HWC',
                    )
                except Exception as e:
                    logger.info(f'Failed to render episode:{e}')
                    pass

    def reset(self) -> None:
        pass

    def get(self) -> Mapping[Text, float]:
        return {}
class TensorboardLearnerStatisticsTracker:
    """Write learner statistics to tensorboard, for parallel training agents with actor-learner scheme"""

    def __init__(self, writer: SummaryWriter):
        self._total_steps = 0  # keep track total number of steps, does not reset
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()
        self._writer = writer

    def step(self, stats) -> None:

        self._total_steps += 1
        self._num_steps_since_reset += 1

        # Log every N learner steps.
        if self._total_steps % 100 == 0:
            time_stats = self.get()
            self._writer.add_scalar('learner_statistics(learner_steps)/step_rate', time_stats['step_rate'], self._total_steps)

            # This should not block the training loop if there's any exception.
            try:
                if stats:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f'learner_statistics(learner_steps)/{k}', v, self._total_steps)
            except Exception:
                pass

    def reset(self) -> None:
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def get(self) -> Mapping[Text, float]:
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps_since_reset': self._num_steps_since_reset,
            'duration': duration,
        }


def make_default_trackers(log_dir=None, debug_screenshots_interval=0,folder=None):
    """
    Create trackers for the training/evaluation run.

    Args:
        log_dir: tensorboard runtime log directory.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard, default 0 no screenshots.
    """

    if log_dir:
        if folder is not None:
            log_dir = Path(folder + f'/runs/{log_dir}')
        else:
            log_dir = Path(f'runs/{log_dir}')
        # Remove existing log directory
        if log_dir.exists() and log_dir.is_dir():
            shutil.rmtree(log_dir)

        writer = SummaryWriter(log_dir)

        trackers = [
            TensorboardEpisodeTracker(writer),
            TensorboardStepRateTracker(writer),
            TensorboardAgentStatisticsTracker(writer),
        ]

        if debug_screenshots_interval > 0:
            trackers.append(TensorboardScreenshotTracker(writer, debug_screenshots_interval))

        return trackers

    else:
        return [EpisodeTracker(), StepRateTracker()]
    

def make_learner_trackers(run_log_dir=None,folder=None):
    """
    Create trackers for learner for parallel training (actor-learner) run.

    Args:
        run_log_dir: tensorboard run log directory.
    """

    if run_log_dir:
        if folder is not None:
            run_log_dir = Path(folder + f'/runs/{run_log_dir}')
        else:
            run_log_dir = Path(f'runs/{run_log_dir}')
        # Remove existing log directory
        if run_log_dir.exists() and run_log_dir.is_dir():
            shutil.rmtree(run_log_dir)

        writer = SummaryWriter(run_log_dir)

        return [TensorboardLearnerStatisticsTracker(writer)]

    else:
        return []


def generate_statistics(
    trackers: Iterable[Any],
    timestep_action_sequence: Iterable[Tuple[Optional[Transition]]],
) -> Mapping[str, Any]:
    """Generates statistics from a sequence of timestep and actions."""
    # Only reset at the start, not between episodes.
    for tracker in trackers:
        tracker.reset()

    for env, timestep_t, agent, a_t in timestep_action_sequence:
        for tracker in trackers:
            tracker.step(env, timestep_t, agent, a_t)

    # Merge all statistics dictionaries into one.
    statistics_dicts = (tracker.get() for tracker in trackers)
    return dict(collections.ChainMap(*statistics_dicts))