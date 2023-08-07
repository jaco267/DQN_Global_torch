import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple
import random
from Trainer.algos.segment_tree import MinSegmentTree, SumSegmentTree
class ReplayBuffer:
  """A simple numpy replay buffer."""
  def __init__(self, obs_dim: int, size: int=50000, batch_size: int = 32):
    self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size], dtype=np.float32)
    self.rews_buf = np.zeros([size], dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.max_size, self.batch_size = size, batch_size
    self.ptr, self.size, = 0, 0
  def can_sample(self) -> bool:
     return self.size > self.batch_size
  def store(self,
        obs: np.ndarray,act: np.ndarray, 
        rew: float,  next_obs: np.ndarray,  done: bool,
    ):
    self.obs_buf[self.ptr] = obs
    self.next_obs_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)
  def sample_batch(self) -> Dict[str, np.ndarray]:
    idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
    return dict(obs=self.obs_buf[idxs],
    			next_obs=self.next_obs_buf[idxs],
    			acts=self.acts_buf[idxs],
    			rews=self.rews_buf[idxs],
    			done=self.done_buf[idxs])
  def __len__(self) -> int:
    return self.size

class PER(ReplayBuffer):
    """Prioritized Replay buffer.
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
    """
    def __init__(self, obs_dim: int, size: int=50000,  batch_size: int = 32, alpha: float = 0.6):
        """Initialization."""
        assert alpha >= 0
        super().__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)  #* we can find sum in any segment in logN time (instead of N)
        self.min_tree = MinSegmentTree(tree_capacity)  #* we can find min in any segment in logN time 
        
    def store( self, 
        obs: np.ndarray,  act: int, 
        rew: float,       next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha   #0.6 
         #a^0.6 < a  if a == 1 a^0,6 = 1
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        indices = self._sample_proportional()
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,  next_obs=next_obs,
            acts=acts, rews=rews,
            done=done,   weights=weights,
            indices=indices,
        )  
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions.
        Pi = Pi / sum(pi)
        #*  if the value of a node is large, the pr that the upper bound is in there will also be large
        #* Is it possible to sample without replacement?
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)  
        #sum all of the tree values
        # 0, 1 2, 3 4 5 6, ...
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i   #ex.    (a,b)  = (0,5)|(5,10)|(10,15),...
            b = segment * (i + 1)    # upper =   3    6       14 ,...
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)  #*  if the value of a node is large, the pr that the upper bound is in there will also be large
            indices.append(idx)
        return indices
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx.
        importance sampling
        w_i = (1/N* 1/P(i))**beta
        #?? and then it normalize it ???
        """
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()  
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight   #*   0~1

class NstepBuffer:
    def __init__(self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 3, 
        gamma: float = 0.99,
    ):  
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        # doesn't actually store a transition in the buffer, unless `n_step_buffer` is full.
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]
    def can_sample(self) -> bool:
        return self.size > self.batch_size
    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size   


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