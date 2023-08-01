
#!/usr/bin/env python

import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
from torch.nn.utils import clip_grad_norm_
import math
from Trainer.algos.segment_tree import MinSegmentTree, SumSegmentTree
np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(tc.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            tc.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", tc.Tensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(tc.Tensor(out_features))
        self.bias_sigma = nn.Parameter(tc.Tensor(out_features))
        self.register_buffer("bias_epsilon", tc.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    def forward(self, x: tc.Tensor) -> tc.Tensor:
        """Forward method implementation.
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(  x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    @staticmethod
    def scale_noise(size: int) -> tc.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = tc.randn(size)
        return x.sign().mul(x.abs().sqrt())
class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size):
    super().__init__()
    lay = [32,64,32]
    self.nn1 = nn.Linear(obs_size,lay[0])
    self.nn2 = nn.Linear(lay[0],lay[1])
    # set advantage layer
    self.advantage_hidden_layer = NoisyLinear(lay[1],lay[2])
    self.advantage_layer = NoisyLinear(lay[2], action_size)
    # set value layer
    self.value_hidden_layer = NoisyLinear(lay[1], lay[2])
    self.value_layer = NoisyLinear(lay[2],1)
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    
    x = x.reshape([batch_size,-1])
    x = F.relu(self.nn1(x))
    x = F.relu(self.nn2(x))

    val_hid = F.relu(self.value_hidden_layer(x))
    adv_hid = F.relu(self.advantage_hidden_layer(x))

    value = self.value_layer(val_hid)
    advantage = self.advantage_layer(adv_hid)

    q = value + advantage - advantage.mean(dim=-1, keepdim=True)
    return q
  def reset_noise(self):
    self.advantage_hidden_layer.reset_noise()
    self.advantage_layer.reset_noise()
    self.value_hidden_layer.reset_noise()
    self.value_layer.reset_noise()  

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
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
    """
    def __init__(self, obs_dim: int, max_size: int=50000,  batch_size: int = 32, 
        alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0
        super().__init__(obs_dim, max_size, batch_size)
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

        another weird chose is the segment,
        it's a little like sample without replacement  
          #* I think gumbel will be better?
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

class DQN_Agent(): 
  def __init__(self,  gridgraph, self_play_episode_num=20):
    print("----noisy_agent---")
    self.env = gridgraph
    action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    self.gamma = 0.95
    self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
    self.batch_size = 32
    self.dqn = QNetwork(obs_size,action_size).to(device)
    self.dqn_target = QNetwork(obs_size,action_size).to(device)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()
    #** PER
    self.beta = 0.6   #* from 0.6 to 1 # importance sampling at the end   
    alpha = 0.2   #* temperature  (lower the alpha, closer to uniform)
    self.prior_eps = 1e-6 #* guarentees every transition can be sampled
    self.memory = PrioritizedReplayBuffer(obs_dim=obs_size,alpha=alpha)
    #** optim 
    self.criterion = nn.MSELoss(reduction="none")
    self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
  def load_ckpt_or_pass(self,ckpt_folder):
     if os.path.exists(f"{ckpt_folder}model.ckpt"):  #load chpt
          print(f"loading {ckpt_folder}model.ckpt")
          statedict = tc.load(f"{ckpt_folder}model.ckpt")   
          self.dqn.load_state_dict(statedict)
     else:
        print("train from scratch")
  def save_ckpt(self,ckpt_folder):
       tc.save(self.dqn.state_dict(),f"{ckpt_folder}model.ckpt")
  def select_action(self, obs: np.ndarray)->np.ndarray:
    #* dont need eps greedy anymore
    q_values = self.dqn(tc.FloatTensor(obs).to(device=device))  #*  action shape
    selected_action = np.argmax(q_values.cpu().numpy())
    return selected_action
  def step(self,obs, action:np.ndarray)->Tuple[np.float64,bool]:
    nextstate, reward, is_terminal, _ = self.env.step(action) 
    obs_next = self.env.state2obsv() 
    self.memory.store(obs, action, reward, obs_next, is_terminal)   #* save to replay 
    return reward, is_terminal
  def update_model(self) -> None:
    samples = self.memory.sample_batch()  #??self.beta
    weights = tc.FloatTensor(samples["weights"].reshape(-1,1)).to(device) 
    indices = samples['indices']
    elementwise_loss = self._compute_dqn_loss(samples)
    loss = tc.mean(elementwise_loss * weights)  #importance sampling
    self.optim.zero_grad()
    loss.backward()
    clip_grad_norm_(self.dqn.parameters(), 10.0)
    self.optim.step()
    # PER: update priorities
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior + self.prior_eps  
    #* PER larger the loss, larger the priorities
    self.memory.update_priorities(indices, new_priorities)
    # NoisyNet: reset noise
    self.dqn.reset_noise()
    self.dqn_target.reset_noise()
    return
  def _compute_dqn_loss(self,samples: Dict[str,np.ndarray])->tc.Tensor:
    b_obs = tc.FloatTensor(samples["obs"]).to(device)
    b_action =tc.LongTensor(samples["acts"].reshape(-1, 1)).to(device)  # action
    b_reward = tc.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)  # reward
    b_obs_next = tc.FloatTensor(samples["next_obs"]).to(device)
    b_done = tc.FloatTensor(samples["done"].reshape(-1, 1)).to(device) # is term
    
    q_batch = self.dqn(b_obs)#.gather(1,b_action)   
          # action generate by q_batch
    with tc.no_grad():  #* trainable false
      # q_batch_next = self.dqn_target(b_obs_next).max(dim=1,keepdim=True)[0].detach()
      q_batch_next = self.dqn_target(b_obs_next).gather(
          1,self.dqn(b_obs_next).argmax(dim=1,keepdim=True)
      )
      y_batch = (b_reward+self.gamma*q_batch_next*(1-b_done))#.to(device) 
      targetQ = q_batch.cpu()
      # print(targetQ.shape,b_action.shape,y_batch.shape,"sss")
      targetQ[tc.arange(self.batch_size),b_action.cpu().reshape(-1)] = y_batch.cpu().reshape(-1)
    # print(q_batch.shape,targetQ.shape)  #(32,6)
    elementwise_loss = tc.mean(self.criterion(q_batch,targetQ.to(device)),dim=-1)  #* no mean
    return elementwise_loss
  def train(self,
	   	twoPinNumEachNet,  # len = netNum in one file = 20 , ex. [3, 2, 2, 1, 4, 2, 3,..., 4]
	   	netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
	   	ckpt_folder,  #"../model_(train/test)"   #model will be saved to ../model/
  ):        
    self.load_ckpt_or_pass(ckpt_folder)
    results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
    twoPinNum = len(self.env.twopin_combo)#ex. 49,20net has 49 pin connect, avg_connect/net ~=2.5
    update_count = 0;   frames_i = 0
    num_frames = self.max_episodes*twoPinNum
    print("len memory",len(self.memory))
    for episode in range(self.max_episodes):	
      for pin in range(twoPinNum):
        frames_i+=1
        #* PER: increase beta (importance sampling)
        fraction = min(frames_i/num_frames,1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)  #beta ~ 1
        state, reward_plot, is_best = self.env.reset(self.max_episodes)     #*  New loop!
        reward_plot_pure = reward_plot-self.env.posTwoPinNum*100
        if (episode) % twoPinNum == 0:      #*  after one episode
            results['reward_plot_combo'].append(reward_plot)
            results['reward_plot_combo_pure'].append(reward_plot_pure)
        is_terminal = False;     rewardfortwopin = 0
        while not is_terminal:
          obs = self.env.state2obsv()
          with tc.no_grad():
            action = self.select_action(obs)
            reward,is_terminal = self.step(obs,action)
            rewardfortwopin += reward   
          if len(self.memory) >= self.batch_size:
            update_count+=1
            self.update_model()
            if update_count % 100 == 0:
               self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.env.instantrewardcombo.append(rewardfortwopin)
    print('\nSave model')
    self.save_ckpt(ckpt_folder)
    
    print(f"Model saved in path: {ckpt_folder}")
    solution = self.env.best_route[-twoPinNum:]
		
    for i in range(len(netSort)):
        results['solutionDRL'].append([])
    if self.env.posTwoPinNum  == twoPinNum:
        dumpPointer = 0
        for i in range(len(netSort)):
            netToDump = netSort[i]
            for j in range(twoPinNumEachNet[netToDump]):
                results['solutionDRL'][netToDump].append(solution[dumpPointer])
                dumpPointer = dumpPointer + 1
    else:
        results['solutionDRL'] = solution
			
    return results,	 solution,  self.env.posTwoPinNum




