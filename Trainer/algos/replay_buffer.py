import numpy as np
from typing import Dict, List, Tuple
import random

class Episode_ReplayBuffer:
  """A simple numpy replay buffer."""
  def __init__(self,  obs_dim: int=12, frame_size: int=50000, batch_size: int = 32,   context_len=5,  env_max_steps =  100  #* gridword max steps
  ):
    self.context_len = context_len;  self.obs_dim = obs_dim
    self.frame_size = frame_size;  self.batch_size =  batch_size
    self.episode_max_size = frame_size // env_max_steps  #** 500
    self.env_max_steps = env_max_steps
    # Keeps first and last obs together for +1
    self.obs_buf = np.zeros([self.episode_max_size,env_max_steps+1,obs_dim], dtype = np.float32)
    # action (eps,frame_per_eps+1,1) = (2000, 101, 1)#* because we may want to predict action
    self.acts_buf = np.zeros([self.episode_max_size,env_max_steps+1,1],dtype=np.uint8) 
    self.rews_buf = np.zeros([self.episode_max_size,env_max_steps,1],dtype=np.float32)
    self.done_buf = np.zeros([self.episode_max_size,env_max_steps,1], dtype=np.bool_)
    self.episode_len = np.zeros([self.episode_max_size], dtype=np.uint8)
    self.ptr = [0,0]; self.size = 0;
  def store(self,  obs: np.ndarray,act: np.ndarray, rew: float,  done: bool, episode_len):
    episode_idx = self.ptr[0] % self.episode_max_size
    obs_idx = self.ptr[1]
    # because in the beginning we already have initial obs state
    self.obs_buf[episode_idx,obs_idx+1] = obs
    self.acts_buf[episode_idx, obs_idx] = act  
    self.rews_buf[episode_idx, obs_idx] = rew
    self.done_buf[episode_idx, obs_idx] = done
    self.episode_len[episode_idx] = episode_len  
    self.ptr = [self.ptr[0], self.ptr[1] + 1]
    self.size = min(self.size + 1, self.frame_size)
  def initialize_episode_buffer(self, obs: np.ndarray) -> None:  
    """Use this at the beginning of the episode to store_ the first obs"""
    episode_idx = self.ptr[0] % self.episode_max_size
    self.initialize_episode(episode_idx)  #*  reset episode
    self.obs_buf[episode_idx, 0] = obs
  def can_sample(self) -> bool:
      return self.batch_size < self.ptr[0]
  def point_to_next_episode(self):  
      self.ptr = [self.ptr[0] + 1, 0]
  def initialize_episode(self, episode_idx: int) -> None:  #* reset episode
      # Cleanse the episode of any previous data
      self.obs_buf[episode_idx]=np.zeros([self.env_max_steps+1, self.obs_dim],dtype=np.float32)
      self.acts_buf[episode_idx] = np.zeros([self.env_max_steps + 1, 1], dtype=np.uint8)
      self.rews_buf[episode_idx] = np.zeros([self.env_max_steps,1], dtype=np.float32)
      self.done_buf[episode_idx] = np.ones([self.env_max_steps, 1], dtype=np.bool_)
      self.episode_len[episode_idx] = 0
  def sample_batch(self) -> Dict[str, np.ndarray]:
    # Exclude the current episode we're in  # all episode before current episode can be sampled 
    valid_episodes = [i for i in range(min(self.ptr[0], self.episode_max_size))
        if i != self.ptr[0] % self.episode_max_size] #todo sample without replacement???
    episode_idxes = np.array([[random.choice(valid_episodes)] for _ in range(self.batch_size)])
    #episode_idxes = [[ep10],[ep51],..,[ep2]]
    #* sample random seq_len slices from random episode
    transition_starts=np.array([random.randint(0,max(0,self.episode_len[idx[0]]-self.context_len))
            for idx in episode_idxes])  #ex. idx == [ep51]
    transitions = np.array([range(start, start + self.context_len) for start in transition_starts])
    return dict(
       obs=self.obs_buf[episode_idxes,transitions],
       acts=self.acts_buf[episode_idxes,transitions], rews=self.rews_buf[episode_idxes,transitions],
   next_obs=self.obs_buf[episode_idxes,transitions+1],done=self.done_buf[episode_idxes,transitions],
    ctx_len=np.clip(self.episode_len[episode_idxes],0,self.context_len)   
        )
  def __len__(self) -> int:  return self.size