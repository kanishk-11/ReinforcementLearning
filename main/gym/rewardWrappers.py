
import gymnasium as gym


class BasicGymWrapper(gym.Wrapper):
    def step(self,action):
        observation,reward,terminated,truncated,info = self.env.step(action)
        info['raw_reward'] = reward
        return observation,reward,terminated,truncated,info

class RewardClipper(gym.RewardWrapper):
    def __init__(self,env,reward_clip):
        super().__init__(env)
        self.reward_clip = abs(reward_clip)
    def reward(self,reward):
        return None if reward is None else max(min(reward,self.reward_clip),-self.reward_clip)
        
