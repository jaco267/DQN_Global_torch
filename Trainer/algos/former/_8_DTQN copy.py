'''
#https://github.com/kevslinger/DTQN
@article{esslinger2022dtqn,
  title = {Deep Transformer Q-Networks for Partially Observable Reinforcement Learning},
  author = {Esslinger, Kevin and Platt, Robert and Amato, Christopher},
  journal= {arXiv preprint arXiv:2206.01078},
  year = {2022},
}
'''
#!/usr/bin/env python

import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

class TransformerLayer(nn.Module):
    """Create a single transformer block. DTQN may stack multiple blocks."""
    def __init__( self,
        num_heads: int,#  Number of heads to use for MultiHeadAttention.
        embed_dim: int,#  The dimensionality of the layer.
        history_len: int,# The maximum number of observations to take in.
        dropout: float=0,#  Dropout percentage.
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(  
            embed_dim=embed_dim,num_heads=num_heads,  
            dropout=dropout,    batch_first=True
        )
        self.ffn = nn.Sequential( 
            nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout),
        )
        # Just storage for attention weights for visualization
        self.alpha = None
        # Set up causal masking for attention
        self.attn_mask = nn.Parameter(
            tc.triu(tc.ones(history_len, history_len), diagonal=1),
            requires_grad=False,
        )
        self.attn_mask[self.attn_mask.bool()] = -float("inf")
        """
          The mask will look like:
          [0, -inf, -inf, ..., -inf]
          [0,    0, -inf, ..., -inf]
          ...
          [0,    0,    0, ...,    0]
          Where 0 means that timestep is allowed to attend.
          So the first timestep can attend only to the first timestep
          and the last timestep can attend to all observations.        
        """
    def forward(self, x: tc.Tensor) -> tc.Tensor:
        # bs, seq_len, dim
        attention, self.alpha = self.attention(
            x,x,x,  attn_mask=self.attn_mask[: x.size(1), : x.size(1)],
            average_attn_weights=True,  # Only affects self.alpha for visualizations
        )
        # Skip connection then LayerNorm
        x = x + F.relu(attention)
        x = self.layernorm1(x)
        ffn = self.ffn(x)
        # Skip connection then LayerNorm
        x = x + F.relu(ffn)
        x = self.layernorm2(x)
        return x
def init_weights(module):
    # print(module,"????")
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.in_proj_bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size,embed_dim = 128,context_len=50,num_heads=8): #* 128 is a little too large...
    super().__init__()
    # lay = [32,64,32]
    self.position_embedding = nn.Parameter(tc.zeros(1,context_len,embed_dim),requires_grad=True)
    self.obs_embedding = nn.Linear(obs_size, embed_dim)
    self.dropout = nn.Dropout(0) #todo
    num_layers = 3
    self.transformer_layers = nn.Sequential(
        *[  TransformerLayer(num_heads=num_heads, embed_dim=embed_dim, history_len=context_len)
            for _ in range(num_layers)
         ]
    )
    self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_size),
        )
    self.history_len = context_len
    #  Applies ``fn`` recursively to every submodule 
    self.apply(init_weights)
    # self.nn1 = nn.Linear(obs_size,lay[0])
    # self.nn2 = nn.Linear(lay[0],lay[1])
    # self.nn3 = nn.Linear(lay[1],lay[2])
    # self.nn4 = nn.Linear(lay[2],action_size)
  def forward(self,obs):
    # x (bs,seq_len,obs_dim) 
    bs,seq_len,obs_dim = obs.shape
    obs= tc.flatten(obs,start_dim=0,end_dim=1)  #(bs*seq_len,obs_dim) = (bs*l,12)
    obs_embed = self.obs_embedding(obs)  #(bs*seq_len,outer_embed_size)
    obs_embed = obs_embed.reshape(bs,seq_len,obs_embed.size(-1)) #(bs,seq_len,outer_embed_size)
    working_memory = self.transformer_layers(obs_embed)
    output = self.ffn(working_memory)
    return output[:, -(seq_len):, :] #[32,50,6]
class ReplayBuffer:
  """A simple numpy replay buffer."""
  def __init__(self, 
        obs_dim: int=12, frame_size: int=50000, 
        batch_size: int = 32,
        context_len=50,
        max_episode_steps =  100  #* gridword max steps
  ):
    self.frame_size = frame_size
    self.episode_max_size = frame_size // max_episode_steps
    self.batch_size =  batch_size
    self.context_len = context_len
    self.obs_dim = obs_dim
    self.max_episode_steps = max_episode_steps
    self.ptr = [0,0]; self.size = 0;
    # Keeps first and last obs together for +1
    self.obs_buf = np.zeros([self.episode_max_size,max_episode_steps+1,obs_dim], dtype = np.float32)
    #* we dont need next_obs_buf now
    # action (eps,frame_per_eps+1,1) = (2000, 101, 1)#* because we may want to predict action
    self.acts_buf = np.zeros([self.episode_max_size,max_episode_steps+1,1],dtype=np.uint8) 
    self.rews_buf = np.zeros([self.episode_max_size,max_episode_steps,1],dtype=np.float32)
    self.done_buf = np.zeros([self.episode_max_size,max_episode_steps,1], dtype=np.bool_)
    self.episode_lengths = np.zeros([self.episode_max_size], dtype=np.uint8)
  def store(self,  obs: np.ndarray,act: np.ndarray, rew: float,  done: bool, episode_len):
    episode_idx = self.ptr[0] % self.episode_max_size
    obs_idx = self.ptr[1]
    # because in the beginning we already have initial obs state
    self.obs_buf[episode_idx,obs_idx+1] = obs
    self.acts_buf[episode_idx, obs_idx] = act  
    self.rews_buf[episode_idx, obs_idx] = rew
    self.done_buf[episode_idx, obs_idx] = done
    self.episode_lengths[episode_idx] = episode_len   #?????????/
    self.ptr = [self.ptr[0], self.ptr[1] + 1]
    self.size = min(self.size + 1, self.frame_size)
  def initialize_episode_buffer(self, obs: np.ndarray) -> None:  
    """Use this at the beginning of the episode to store the first obs"""
    episode_idx = self.ptr[0] % self.episode_max_size
    self.cleanse_episode(episode_idx)  #*  reset episode
    self.obs_buf[episode_idx, 0] = obs
  def can_sample(self) -> bool:
      return self.batch_size < self.ptr[0]
  def flush(self):   #*** go to next episode
      #todo rename it
      self.ptr = [self.ptr[0] + 1, 0]
  def cleanse_episode(self, episode_idx: int) -> None:  #* reset episode
      #todo rename it
      # Cleanse the episode of any previous data
      #**  (idx,frame_per_eps+1,obs_dim) = (idx, 101, 12) # Keeps first and last obs together for +1
      self.obs_buf[episode_idx]=np.zeros([self.max_episode_steps+1, self.obs_dim],dtype=np.float32)
      #** action (eps,frame_per_eps+1,1) = (2000, 101, 1)
      self.acts_buf[episode_idx] = np.zeros([self.max_episode_steps + 1, 1], dtype=np.uint8)
      self.rews_buf[episode_idx] = np.zeros([self.max_episode_steps,1], dtype=np.float32)
      self.done_buf[episode_idx] = np.ones([self.max_episode_steps, 1], dtype=np.bool_)
      self.episode_lengths[episode_idx] = 0
  def sample_batch(self) -> Dict[str, np.ndarray]:
    # Exclude the current episode we're in
    valid_episodes = [i for i in range(min(self.ptr[0], self.episode_max_size))
        if i != self.ptr[0] % self.episode_max_size]
 # all episode before current episode can be sampled 
    #todo sample without replacement???
    episode_idxes = np.array([[random.choice(valid_episodes)] for _ in range(self.batch_size)])
    #episode_idxes = [[ep10],[ep51],..,[ep2]]
    #* sample random seq_len slices from random episode

    #ex. len(5) ctx = 3   #start = randint(2) = 0 or 1, start+ctx_len= 1+3 = 4, next obs got 4+1
    #that's why we need  zeros(self.max_episode_steps+1)  for obs and action
    
    transition_starts = np.array(
      [random.randint(0, max(0, self.episode_lengths[idx[0]] - self.context_len))
            for idx in episode_idxes])  #ex. idx == [ep51]

    transitions = np.array( 
        [range(start, start + self.context_len) for start in transition_starts])
    return dict(obs=self.obs_buf[episode_idxes,transitions],
                acts=self.acts_buf[episode_idxes,transitions],
          rews=self.rews_buf[episode_idxes,transitions],
    			next_obs=self.obs_buf[episode_idxes,transitions+1],
    			done=self.done_buf[episode_idxes,transitions],
          ctx_len=np.clip(self.episode_lengths[episode_idxes],0,self.context_len)   #??? 
        )
  def __len__(self) -> int:
    return self.size

class Context:
    """A Dataclass dedicated to storing the agent's history (up to the previous `max_length`)"""
    def __init__( self,
        context_length: int,  # 50 maximum number of transitions to store
        num_actions: int,  # 6
        obs_dim: int,  #12
    ):
        self.max_length = context_length
        self.env_obs_length = obs_dim
        self.num_actions = num_actions
        self.reward_mask = 0.0;        self.done_mask = True
        self.timestep = 0
    def reset(self, obs: np.ndarray):
        """Resets to a fresh context"""
        self.obs = np.zeros([self.max_length, self.env_obs_length])#*(50, 6)
        self.obs[0] = obs  # Initial observation
        rng = np.random.Generator(np.random.PCG64(seed=0))
        self.action = rng.integers(self.num_actions, size=(self.max_length, 1))#*(50,1)
        self.reward = np.full_like(self.action, self.reward_mask)#*(50,1)
        self.done = np.full_like(self.reward, self.done_mask, dtype=np.int32)#*(50,1)
        self.timestep = 0
    def add_transition(self, o: np.ndarray, a: int, r: float, done: bool) -> None:
        """Add an entire transition. If the context is full, evict the oldest transition"""
        self.timestep += 1
        self.obs = self.roll(self.obs)
        self.action = self.roll(self.action)
        self.reward = self.roll(self.reward)
        self.done = self.roll(self.done)
        t = min(self.timestep, self.max_length - 1)
        self.obs[t] = o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])
        return 
    def roll(self, arr: np.ndarray) -> np.ndarray:
        """Utility function to help with insertions at the end of the array. If the context is full, we replace the first element with the new element, then 'roll' the new element to the end of the array"""
        #[0,1,2,3]->[1,2,3,0]
        return np.roll(arr, -1, axis=0) if self.timestep >= self.max_length else arr
class DQN_Agent(): 
    def __init__(self, gridgraph,self_play_episode_num=20,context_len=50):
      print("----DTQN_agent---")
      self.context_len = context_len 
      self.env = gridgraph
      self.action_size = self.env.action_size  #6
      obs_size = self.env.obs_size        #12

      self.dqn = QNetwork(obs_size,self.action_size).to(device)
      self.dqn_target = QNetwork(obs_size,self.action_size).to(device)
      self.dqn_target.load_state_dict(self.dqn.state_dict())
      self.dqn_target.eval()

      self.epsilon = 0.05
      self.gamma = 0.95
      self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
      self.batch_size = 32
      self.replay = ReplayBuffer(obs_dim=obs_size)
      self.grad_norm_clip = 1.0
      self.num_train_steps = 0
      self.target_update_frequency = 100 #update target network every 100 steps
      self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
      self.context = Context(context_len,self.action_size,obs_size)
      self.criterion = nn.MSELoss()
    @tc.no_grad()
    def get_action(self, ):
      # Creating epsilon greedy probabilities to sample from.
      rnd = np.random.rand()
      if rnd <= self.epsilon:   #exploration
        return np.random.randint(self.action_size) #todo self.num_action
      context_obs_tensor = tc.FloatTensor( #50
            self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
        ).unsqueeze(0).to(device)   #todo investigate the shape  so  seq_len range from 1 to 50,  means output of transformer also range from 1 to 50 ??
      q_values = self.dqn(context_obs_tensor)
      return tc.argmax(q_values[:, -1, :]).item()   #argmax(1,-1,action_dim)  #todo  the shape
    def update_model(self,):
      samples = self.replay.sample_batch()       
      b_obs = tc.FloatTensor(samples["obs"]).to(device)
      b_obs_next = tc.FloatTensor(samples["next_obs"]).to(device)
      b_action =tc.LongTensor(samples["acts"]).to(device)  # action
      b_reward = tc.FloatTensor(samples["rews"]).to(device)  # reward
      b_done = tc.FloatTensor(samples["done"]).to(device) # is term
      
      q_batch = self.dqn(b_obs)# action generate by q_batch
      q_values = q_batch.gather(2,b_action).squeeze()
      with tc.no_grad():  #* trainable false
        #  q_batch_next = self.dqn_target(b_obs_next).max(dim=1,keepdim=True)[0].detach()
        # [batch-size x hist-len x n-actions] 
        argmax = tc.argmax(self.dqn(b_obs_next),dim=2).unsqueeze(-1)
        q_batch_next = self.dqn_target(b_obs_next).gather(2,argmax).squeeze()
        # q_batch_next = self.dqn_target(b_obs_next).gather(  #todo change to this
        #   1,self.dqn(b_obs_next).argmax(dim=1,keepdim=True)
        # )
        targets = b_reward.squeeze()+self.gamma*q_batch_next*(1-b_done.squeeze())#.to(device) 
      q_values = q_values[:, -self.context_len :]
      targets = targets[:, -self.context_len :]  
      loss = self.criterion(q_values,targets)
      self.optim.zero_grad()
      loss.backward()
      self.optim.step()
      self.num_train_steps+=1
      if self.num_train_steps % self.target_update_frequency == 0:
          self.dqn_target.load_state_dict(self.dqn.state_dict())
    def train(self,
	   	twoPinNumEachNet,  # len = netNum in one file = 20 , value = netPinNum - 1  ex. [3, 2, 2, 1, 4, 2, 3,..., 4]
	   	netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
	   	savepath,  #"../model_(train/test)"   #model will be saved to ../model/
		  model_file=None  # if model_file = None, training; if given, testing  #* if testing using training function, comment burn_in in Router.py
    ):           
        if os.path.exists(f"{savepath}model.ckpt"):  #load chpt
            print(f"loading {savepath}model.ckpt")
            statedict = tc.load(f"{savepath}model.ckpt")   
            self.dqn.load_state_dict(statedict)
        results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
        twoPinNum = len(self.env.twopin_combo)
        for episode in range(self.max_episodes):	
          for pin in range(twoPinNum):
            state, reward_plot, is_best = self.env.reset(self.max_episodes)     #*  New loop!
            obs = self.env.state2obsv()
            #***   context_reset(env.reset())
            self.context.reset(obs)
            self.replay.initialize_episode_buffer(obs)   
            reward_plot_pure = reward_plot-self.env.posTwoPinNum*100
            if (episode) % twoPinNum == 0:      #*  after one episode
              results['reward_plot_combo'].append(reward_plot)
              results['reward_plot_combo_pure'].append(reward_plot_pure)
            is_terminal = False;    rewardfortwopin = 0
            while not is_terminal:
              with tc.no_grad():
                action = self.get_action()
                nextstate, reward, is_terminal, debug = self.env.step(action)  #* agent step  
                obs = self.env.state2obsv()
                #*update_context_and_buffer
                self.context.add_transition( obs, action, reward, is_terminal )
                self.replay.store(obs,action,reward,is_terminal,self.context.timestep)
                rewardfortwopin = rewardfortwopin + reward   #todo ??
              if self.replay.can_sample():  
                self.update_model()
            self.replay.flush()  #**replay got to next episode
            self.env.instantrewardcombo.append(rewardfortwopin)
        print('\nSave model')
        tc.save(self.dqn.state_dict(),f"{savepath}model.ckpt")
        print(f"Model saved in path: {savepath}")
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





