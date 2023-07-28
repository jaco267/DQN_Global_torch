
#!/usr/bin/env python

import numpy as np
import torch as tc
from collections import deque
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Deque, Dict, List, Tuple
from torch.nn.utils import clip_grad_norm_
import math
from Trainer.algos.segment_tree import MinSegmentTree, SumSegmentTree
import Trainer.algos.resultUtils  as U  #import end_episode
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
        self.register_buffer("weight_epsilon", tc.Tensor(out_features, in_features))
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
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    @staticmethod
    def scale_noise(size: int) -> tc.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = tc.randn(size)

        return x.sign().mul(x.abs().sqrt())
class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size,hid_layer=1,start_end_dim=32,emb_dim=64):
    
    super().__init__()
    # lay = [32,64,64,64,32]  #[32,64,32]
    # self.nn1 = nn.Linear(obs_size,lay[0])
    # self.nn2 = nn.Linear(lay[0],lay[1])
    layers = []
    layers += [nn.Linear(obs_size,start_end_dim), nn.ReLU(),
               nn.Linear(start_end_dim,emb_dim),nn.ReLU(),]
    # hid_len=3 deep   hid_len=1 normal
    for j in range(hid_layer-1):
       layers+=[nn.Linear(emb_dim,emb_dim), nn.ReLU()]
    self.feature_layer = nn.Sequential(
      #  nn.Linear(obs_size,lay[0]), nn.ReLU(),
      #  nn.Linear(lay[0],lay[1]),nn.ReLU(),
      #  nn.Linear(lay[1],lay[2]),nn.ReLU(),
      #  nn.Linear(lay[2],lay[3]),nn.ReLU(),
      *layers
    )
    
    # set advantage layer
    self.advantage_hidden_layer = NoisyLinear(emb_dim,start_end_dim)
    self.advantage_layer = NoisyLinear(start_end_dim, action_size)
    print("deepQ structure",layers,f"adv {emb_dim} {start_end_dim} {action_size}")
    # set value layer
    self.value_hidden_layer = NoisyLinear(emb_dim, start_end_dim)
    self.value_layer = NoisyLinear(start_end_dim,1)
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    
    x = x.reshape([batch_size,-1])
    # x = F.relu(self.nn1(x))
    # x = F.relu(self.nn2(x))
    x = self.feature_layer(x)
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
  def __init__(self, obs_dim: int, size: int=50000, batch_size: int = 32,
               n_step = 1, gamma = 0.99):
    self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size], dtype=np.float32)
    self.rews_buf = np.zeros([size], dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.max_size, self.batch_size = size, batch_size
    self.ptr, self.size, = 0, 0

    #* for N-step Learning
    self.n_step_buffer = deque(maxlen=n_step)
    self.n_step = n_step
    self.gamma = gamma
  def store(self,
        obs: np.ndarray,act: np.ndarray, 
        rew: float,  next_obs: np.ndarray,  done: bool,
    ):
    transition = (obs, act, rew, next_obs, done)
    self.n_step_buffer.append(transition)
    # single step transition is not ready
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

  def sample_batch(self) -> Dict[str, np.ndarray]:
    idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
    return dict(obs=self.obs_buf[idxs],
    			next_obs=self.next_obs_buf[idxs],
    			acts=self.acts_buf[idxs],
    			rews=self.rews_buf[idxs],
    			done=self.done_buf[idxs],
          # for N-step Learning
          indices=idxs,
        )
  def sample_batch_from_idxs(
        self, idxs: np.ndarray
  ) -> Dict[str, np.ndarray]:
    # for N-step Learning
    return dict(
        obs=self.obs_buf[idxs],
        next_obs=self.next_obs_buf[idxs],
        acts=self.acts_buf[idxs],
        rews=self.rews_buf[idxs],
        done=self.done_buf[idxs],
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
        alpha: float = 0.6,n_step = 1, gamma = 0.99
    ):
        """Initialization."""
        assert alpha >= 0
        #*  n_step  init
        super().__init__(obs_dim, max_size, batch_size, n_step, gamma)
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
        transition = super().store(obs, act, rew, next_obs, done)
        if transition:
          self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha   #0.6 
          #a^0.6 < a  if a == 1 a^0,6 = 1
          self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
          self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition
    
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
  def __init__(self,  gridgraph,hid_layer=1,emb_dim=64, self_play_episode_num=20):
    print("----nstep---")
    self.env = gridgraph
    action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    
    self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
    self.batch_size = 32
    self.dqn = QNetwork(obs_size,action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
    self.dqn_target = QNetwork(obs_size,action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()
    self.gamma = 0.95
    #** PER
    self.beta = 0.6   #* from 0.6 to 1 # importance sampling at the end   
    alpha = 0.2   #* temperature  (lower the alpha, closer to uniform)
    self.prior_eps = 1e-6 #* guarentees every transition can be sampled
    #! nstep needs more buffer size
    buffer_size = 100000
    self.memory = PrioritizedReplayBuffer(obs_dim=obs_size,max_size=buffer_size,alpha=alpha,n_step=1)
    #** memory for N-step Learning
    n_step = 3
    self.use_n_step = True if n_step > 1 else False
    if self.use_n_step:
      self.n_step = n_step
      self.memory_n = ReplayBuffer(
          obs_size,size=buffer_size, n_step=n_step, gamma=self.gamma
      )
    self.criterion = nn.MSELoss(reduction="none")
    self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
    #** training results
    self.best_route = []  #solutions
    self.best_reward = 0.0
    self.episode_reward = 0.0
    self.episode_two_pin_rewards = []
    self.PosTwoPinNum = 0#connected pin ex.43/50 32=PosTwoPinNum  50==total_pin=twopin_combo   
    self.route_combo = []
  def select_action(self, obs: np.ndarray)->np.ndarray:
    #* dont need eps greedy anymore
    q_values = self.dqn(tc.FloatTensor(obs).to(device=device))  #*  action shape
    selected_action = np.argmax(q_values.cpu().numpy())
    return selected_action
  def agent_step(self,obs, action:np.ndarray)->Tuple[np.float64,bool]:
    nextstate, reward, is_terminal, _ = self.env.step(action) 
    obs_next = self.env.state2obsv() 
    # N-step transition
    if self.use_n_step:
      one_step_transition = self.memory_n.store(obs, action, reward, obs_next, is_terminal)   #* save to replay 
    else:
       one_step_transition = (obs, action, reward, obs_next, is_terminal)
    if one_step_transition:
       self.memory.store(*one_step_transition)
    return reward, is_terminal
  def update_model(self) -> None:
    samples = self.memory.sample_batch()           #!!!!!!!!
    weights = tc.FloatTensor(samples["weights"].reshape(-1,1)).to(device) 
    indices = samples['indices']
    elementwise_loss = self._compute_dqn_loss(samples,self.gamma)
    # loss = tc.mean(elementwise_loss * weights)  #importance sampling
    # N-step Learning loss
    if self.use_n_step:
      gamma = self.gamma ** self.n_step
      samples = self.memory_n.sample_batch_from_idxs(indices)
      elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
      elementwise_loss += elementwise_loss_n_loss
      
      # PER: importance sampling before average
      loss = tc.mean(elementwise_loss * weights)

    self.optim.zero_grad()
    loss.backward()
    clip_grad_norm_(self.dqn.parameters(), 10.0)
    self.optim.step()
    # PER: update priorities
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior + self.prior_eps  
    #*  larger the loss, larger the priorities
    self.memory.update_priorities(indices, new_priorities)
    # NoisyNet: reset noise
    self.dqn.reset_noise()
    self.dqn_target.reset_noise()
    return
  def _compute_dqn_loss(self,samples: Dict[str,np.ndarray],gamma)->tc.Tensor:
    state = tc.FloatTensor(samples["obs"]).to(device)
    action =tc.LongTensor(samples["acts"]).reshape(-1,1).to(device)  # action
    reward = tc.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)  # reward
    next_state = tc.FloatTensor(samples["next_obs"]).to(device)
    done = tc.FloatTensor(samples["done"].reshape(-1, 1)).to(device) # is term
    
    curr_q_value = self.dqn(state).gather(1,action)   
    with tc.no_grad():  
      next_q_value = self.dqn_target(next_state).gather(
          1,self.dqn(next_state).argmax(dim=1,keepdim=True)
      )
      mask = 1-done
      target = (reward+gamma*next_q_value*mask).to(device) 
    elementwise_loss = self.criterion(curr_q_value.reshape(-1),target.reshape(-1))  #* no mean
    return elementwise_loss

  def train(self,
	   	twoPinNumEachNet,  # len = netNum in one file = 20 , ex. [3, 2, 2, 1, 4, 2, 3,..., 4]
	   	netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
	   	ckpt_path,  #"../model_(train/test)"   #model will be saved to ../model/
      logger, #wandb
      save_ckpt=False,
      load_ckpt=True,
      early_stop=False,
  ):        
    print(">>>>>>>>>>\n",locals())
    if load_ckpt:
      U.load_ckpt_or_pass(self,ckpt_path)
    results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
    twoPinNum = len(self.env.twopin_combo)#ex. 49,20net has 49 pin connect, avg_connect/net ~=2.5
    update_count = 0;   frames_i = 0
    num_frames = self.max_episodes*twoPinNum
    reward=0
    for episode in range(self.max_episodes):	
      U.init_episode(self)
      for pin in range(twoPinNum):
        # if print_reward:
        #     print(f"{pin}/{twoPinNum}      reward={reward}\r",end="")
        frames_i+=1
        #* PER: increase beta (importance sampling)
        fraction = min(frames_i/num_frames,1.0)
        
        self.beta = self.beta + fraction * (1.0 - self.beta)  #beta ~ 1
        state = self.env.reset(pin)     
        is_terminal = False;     
        rewardfortwopin = 0
        while not is_terminal:
          obs = self.env.state2obsv()
          with tc.no_grad():
            action = self.select_action(obs)
            reward,is_terminal = self.agent_step(obs,action)
            self.episode_reward = self.episode_reward + reward     #!!!
            rewardfortwopin += reward   
          if len(self.memory) >= self.batch_size:
            update_count+=1
            self.update_model()
            if update_count % 100 == 0:
               self.dqn_target.load_state_dict(self.dqn.state_dict())
        U.update_pin_result(self,rewardfortwopin)
      #*  finished one episode  deep reset env
      U.end_episode(self,logger,episode,results)
      if early_stop == True:  #pre-training mode 
         if self.PosTwoPinNum/len(self.env.twopin_combo) > 0.9:
            print("early stopping when training to prevent overfitting")
            break      
    if save_ckpt:
      print('\nSave model')
      tc.save(self.dqn.state_dict(),ckpt_path)
      print(f"Model saved in path: {ckpt_path}")
    else:
      print("evaluating   dont save model")
    solution = self.best_route[-twoPinNum:]
		
    for i in range(len(netSort)):  #20
        results['solutionDRL'].append([])
    success = 0
    if self.PosTwoPinNum  == twoPinNum:
      dumpPointer = 0
      for i in range(len(netSort)):
          netToDump = netSort[i]
          for j in range(twoPinNumEachNet[netToDump]):
              results['solutionDRL'][netToDump].append(solution[dumpPointer])
              dumpPointer = dumpPointer + 1
      success = 1
    else:
        results['solutionDRL'] = solution
			
    return results,	 solution,  self.PosTwoPinNum,success





