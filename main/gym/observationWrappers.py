import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import cv2
class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self,env,modification_probability):
        super().__init__(env)
        if not 0.0<=modification_probability<1.0:
            raise ValueError("modification probability must be between 0.0 and 1.0")
        self.modification_probability = modification_probability
    def observation(self,observation):
        if self.env.unwrapped.np_random.random()<=self.modification_probability:
            observation_new = np.zeros_like(observation,dtype=self.observation_space.dtype)
        return observation_new

class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grascale image as done in the Nature paper.
    """

    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)

        assert self.observation_space.dtype == np.uint8 and len(self.observation_space.shape) == 3

        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.uint8,
        )

    def observation(self, obs):

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = np.expand_dims(obs, -1)

        return obs

class ScaleFrame(gym.ObservationWrapper):
    """Scale frame by divide 255."""

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0

class ObservationChannelFirst(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env:gym.Env, scale_obs):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        _low, _high = (0.0, 255) if not scale_obs else (0.0, 1.0)
        new_dtype = env.observation_space.dtype if not scale_obs else np.float32
        self.observation_space = Box(low=_low, high=_high, shape=new_shape, dtype=new_dtype)

    def observation(self, obs):
        # permute [H, W, C] array to in the range [C, H, W]
        # return np.transpose(observation, axes=(2, 0, 1)).astype(self.observation_space.dtype)
        obs = np.asarray(obs, dtype=self.observation_space.dtype).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        return np.ascontiguousarray(obs, dtype=self.observation_space.dtype)
class ObservationToNumpy(gym.ObservationWrapper):
    """Make the observation into numpy ndarrays."""

    def observation(self, obs):
        return np.asarray(obs, dtype=self.observation_space.dtype)