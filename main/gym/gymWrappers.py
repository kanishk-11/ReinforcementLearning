
from collections import deque
from symbol import term
import gymnasium as gym
import numpy as np


class NoopStart(gym.Wrapper):
    def __init__(self,env:gym.Env,steps=50):
        gym.Wrapper.__init__(self,env)
        self.noop_steps = steps
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0]=='NOOP'
    def reset(self,**kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(0,self.noop_steps) + 1
        
        for noop_timestep in range(noops):
            obs, _ , truncated ,done, info = self.env.step(self.noop_action)
            if done or truncated:
                obs = self.env.reset(**kwargs)
            return obs,info
    
    def step(self,action):
        return self.env.step(action)

class StickyAction(gym.Wrapper):
    def __init__(self,env,sticky_action_probablility = 0.25):
        gym.Wrapper.__init__(self,env)
        self.sticky_action_probablility = sticky_action_probablility
        self.last_action = 0
    def step(self,action):
        if np.random.uniform() < self.sticky_action_probablility:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)
    
    def reset(self,**kwargs):
        self.last_action=0
        return self.env.reset(**kwargs)

class FrameSkip(gym.Wrapper):
    def __init__(self,env:gym.Env,frame_skip=4):
        gym.Wrapper.__init__(self,env)
        self.frame_skip_num = frame_skip
        self._observation_buffer = np.zeros((2,)+(env.observation_space.shape),dtype= np.uint8)
    def step(self,action):
        total_reward = 0.0
        done = None
        for i in range(self.frame_skip_num):
            """Take last 2 frames, get max based on that"""
            obs,reward,terminated,truncated,info = self.env.step(action)
            done = terminated or truncated
            if i == self.frame_skip_num-2:
                self._observation_buffer[0] = obs
            elif i == self.frame_skip_num-1:
                self._observation_buffer[1] = obs
            total_reward += reward
            if done: 
                break
        max_frame = self._observation_buffer.max(axis=0)
        return max_frame,total_reward,terminated,truncated,info
    def reset(self,**kwargs):
        return self.env.reset(**kwargs)
    
class LifeLoss(gym.Wrapper):
    def __init__(self,env:gym.Env):
        gym.Wrapper.__init__(self,env)
        self.lives = 0
        self.really_terminated = True
    
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        self.really_terminated = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            info['loss_life'] = True
        else:
            info['loss_life'] = False
        self.lives = lives
        return obs,reward,self.really_terminated,truncated,info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.really_terminated:
            obs ,info= self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _,_, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs,info

class FrameStack(gym.Wrapper):
    def __init__(self,env,frames_to_stack):
        gym.Wrapper.__init__(self,env)
        self.frames_to_stack = frames_to_stack
        self.frames = deque([],maxlen=frames_to_stack)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=(shape[:-1] + (shape[-1]*self.frames_to_stack,)),dtype=self.observation_space.dtype)
    def reset(self, **kwargs):
        obs ,info = self.env.reset(**kwargs)
        for _ in range(self.frames_to_stack):
            self.frames.append(obs)
        return self._get_obs(),info
    def step(self, action):
        obs, reward, terminated,truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated,truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.frames_to_stack
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
    
class VisitedRoomInfo(gym.Wrapper):
    """Add number of unique visited rooms to the info dictionary.
    For Atari games like MontezumaRevenge and Pitfall.
    """

    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = self.env.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            info['episode_visited_rooms'] = len(self.visited_rooms)
            self.visited_rooms.clear()
        return obs, rew, done, info