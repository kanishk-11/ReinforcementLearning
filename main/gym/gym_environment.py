
import gymnasium as gym

from main.gym.gymWrappers import FrameSkip, FrameStack, LifeLoss, NoopStart, StickyAction, VisitedRoomInfo
from main.gym.observationWrappers import ObservationChannelFirst, ObservationToNumpy, POMDPWrapper, ResizeAndGrayscaleFrame, ScaleFrame
from main.gym.rewardWrappers import BasicGymWrapper, RewardClipper
CLASSIC_ENV_NAMES = ['CartPole-v1', 'LunarLander-v2', 'MountainCar-v0', 'Acrobot-v1']


def get_classic_environment(
    env_name: str,
    seed: int = 1,
    max_abs_reward: int = None,
    obscure_epsilon: float = 0.0,
) -> gym.Env:
    env = gym.make(env_name,render_mode ="rgb_array")
    if max_abs_reward is not None:
        env = BasicGymWrapper(env)
        env = RewardClipper(env, abs(max_abs_reward))

    if obscure_epsilon > 0.0:
        env = POMDPWrapper(env, obscure_epsilon)

    return env

def get_atari_environment(
    env_name: str,
    seed: int = 1,
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    obscure_epsilon: float = 0.0,
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    sticky_action: bool = True,
    scale_obs: bool = False,
    channel_first: bool = True,
) -> gym.Env:
    if 'NoFrameSkip' in env_name:
        raise ValueError('env_name should not include NoFrameSkip')
    env = gym.make(f'{env_name}NoFrameskip-v4',render_mode = "rgb_array")
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=None if max_episode_steps <= 0 else max_episode_steps)
    env.seed(seed)
    if noop_max > 0:
        env = NoopStart(env)
    if sticky_action:
        env = StickyAction(env)
    
    if frame_skip > 0:
        env = FrameSkip(env, frame_skip)

    if obscure_epsilon > 0.0:
        env = POMDPWrapper(env,obscure_epsilon)
    
    if terminal_on_life_loss:
        env = LifeLoss(env)
    
    env = ResizeAndGrayscaleFrame(env, width=frame_width, height=frame_height)

    if scale_obs:
        env = ScaleFrame(env)
    
    if clip_reward:
        env = BasicGymWrapper(env)
        env = RewardClipper(env, 1.0)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if channel_first:
        env = ObservationChannelFirst(env, scale_obs)
    else:
        # This is required as LazeFrame object is not numpy.array.
        env = ObservationToNumpy(env)

    if 'Montezuma' in env_name or 'Pitfall' in env_name:
        env = VisitedRoomInfo(env, room_address=3 if 'Montezuma' in env_name else 1)

    return env