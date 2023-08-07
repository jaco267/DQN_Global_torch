'''
#https://github.com/kevslinger/DTQN
@article{esslinger2022dtqn,
  title = {Deep Transformer Q-Networks for Partially Observable Reinforcement Learning},
  author = {Esslinger, Kevin and Platt, Robert and Amato, Christopher},
  journal= {arXiv preprint arXiv:2206.01078},
  year = {2022},
}
'''
import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
from Trainer.algos.transformer_utils.transformer import TransformerLayer, init_weights, Context
import Trainer.algos.resultUtils  as U
np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size,embed_dim = 64,
               context_len=5,hid_layers=3,num_heads=8): 
    super().__init__()
    self.position_embedding = nn.Parameter(tc.zeros(1,context_len,embed_dim),requires_grad=True)
    self.obs_embedding = nn.Linear(obs_size, embed_dim)
    self.dropout = nn.Dropout(0) #todo
    self.transformer_layers = nn.Sequential(
        *[  TransformerLayer(num_heads=num_heads, embed_dim=embed_dim, history_len=context_len)
            for _ in range(hid_layers)
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
        env_max_steps =  100  #* gridword max steps
  ):
    self.frame_size = frame_size;  self.batch_size =  batch_size
    self.context_len = context_len;    self.obs_dim = obs_dim
    self.episode_max_size = frame_size // env_max_steps
    self.env_max_steps = env_max_steps
    # Keeps first and last obs together for +1
    self.obs_buf = np.zeros([self.episode_max_size,env_max_steps+1,obs_dim], dtype = np.float32)
    #* we dont need next_obs_buf now
    # action (eps,frame_per_eps+1,1) = (2000, 101, 1)#* because we may want to predict action
    self.acts_buf = np.zeros([self.episode_max_size,env_max_steps+1,1],dtype=np.uint8) 
    self.rews_buf = np.zeros([self.episode_max_size,env_max_steps,1],dtype=np.float32)
    self.done_buf = np.zeros([self.episode_max_size,env_max_steps,1], dtype=np.bool_)
    self.episode_lengths = np.zeros([self.episode_max_size], dtype=np.uint8)
    self.ptr = [0,0]; self.size = 0;
  def store(self,  obs: np.ndarray,act: np.ndarray, rew: float,  done: bool, episode_len):
    episode_idx = self.ptr[0] % self.episode_max_size
    obs_idx = self.ptr[1]
    # because in the beginning we already have initial obs state
    self.obs_buf[episode_idx,obs_idx+1] = obs
    self.acts_buf[episode_idx, obs_idx] = act  
    self.rews_buf[episode_idx, obs_idx] = rew
    self.done_buf[episode_idx, obs_idx] = done
    self.episode_lengths[episode_idx] = episode_len   
    self.ptr = [self.ptr[0], self.ptr[1] + 1]
    self.size = min(self.size + 1, self.frame_size)
  def initialize_episode_buffer(self, obs: np.ndarray) -> None:  
    """Use this at the beginning of the episode to store the first obs"""
    episode_idx = self.ptr[0] % self.episode_max_size
    self.initialize_episode(episode_idx)  #*  reset episode
    self.obs_buf[episode_idx, 0] = obs
  def can_sample(self) -> bool:
      return self.ptr[0] > self.batch_size  
  def point_to_next_episode(self):   #*** go to next episode
      self.ptr = [self.ptr[0] + 1, 0]
  def initialize_episode(self, episode_idx: int) -> None:  #* reset episode
      # Cleanse the episode of any previous data
      #**  (idx,frame_per_eps+1,obs_dim) = (idx, 101, 12) # Keeps first and last obs together for +1
      self.obs_buf[episode_idx]=np.zeros([self.env_max_steps+1, self.obs_dim],dtype=np.float32)
      #** action (eps,frame_per_eps+1,1) = (2000, 101, 1)
      self.acts_buf[episode_idx] = np.zeros([self.env_max_steps + 1, 1], dtype=np.uint8)
      self.rews_buf[episode_idx] = np.zeros([self.env_max_steps,1], dtype=np.float32)
      self.done_buf[episode_idx] = np.ones([self.env_max_steps, 1], dtype=np.bool_)
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
    return dict(
      obs=self.obs_buf[episode_idxes,transitions],
      acts=self.acts_buf[episode_idxes,transitions], rews=self.rews_buf[episode_idxes,transitions],
  next_obs=self.obs_buf[episode_idxes,transitions+1],
    	done=self.done_buf[episode_idxes,transitions],
   ctx_len=np.clip(self.episode_lengths[episode_idxes],0,self.context_len)   #??? 
        )
  def __len__(self) -> int:
    return self.size

class DQN_Agent(): 
  def __init__(self, gridgraph,hid_layer,emb_dim,
               self_play_episode_num=150,context_len=5):
    print("----DTQN_epsilon_agent---")
    self.context_len = context_len 
    self.env = gridgraph
    self.action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    env_max_step = self.env.max_step    #100
    self.dqn = QNetwork(obs_size,self.action_size,emb_dim,context_len,hid_layer).to(device)
    self.dqn_target = QNetwork(obs_size,self.action_size,emb_dim,context_len,hid_layer).to(device)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()

    self.replay = ReplayBuffer(
      obs_dim=obs_size,context_len=context_len,env_max_steps=env_max_step)  #100

    self.epsilon = 0.05
    self.gamma = 0.95
    self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
    self.batch_size = 32
    
    self.grad_norm_clip = 1.0
    self.num_train_steps = 0
    self.target_update_frequency = 100 #update target network every 100 steps
    self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
    self.context = Context(context_len,self.action_size,obs_size)
    self.criterion = nn.MSELoss()
    #** training results   (used in resultUtils.py)
    self.result = U.Result() 
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
    loss = self._compute_dqn_loss(samples)    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self.num_train_steps+=1
    if self.num_train_steps % self.target_update_frequency == 0:
        self.dqn_target.load_state_dict(self.dqn.state_dict())
  def _compute_dqn_loss(self,samples: Dict[str,np.ndarray])->tc.Tensor:
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
    return loss

  def train(self,	twoPinNumEachNet,  
	  netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
    ckpt_path,
    logger, save_ckpt=False, load_ckpt= True,
    early_stop=False,
  ):           
    if load_ckpt:                              #, 
      U.load_ckpt_or_pass(self,ckpt_path)
    results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
    twoPinNum = len(self.env.twopin_combo)
    for episode in range(self.max_episodes):	
      self.result.init_episode(agent=self)
      for pin in range(twoPinNum):
        state = self.env.reset(pin) 
        obs = self.env.state2obsv()
        self.context.reset(obs)  #  context_reset(env.reset())
        self.replay.initialize_episode_buffer(obs)   
        is_terminal = False;    rewardfortwopin = 0
        while not is_terminal:
          with tc.no_grad():
            action = self.get_action()
            nextstate, reward, is_terminal,_ = self.env.step(action)  # agent step  
            self.result.update_episode_reward(reward) 
            obs = self.env.state2obsv()
            #*update_context_and_buffer
            self.context.add_transition( obs, action, reward, is_terminal )
            self.replay.store(obs,action,reward,is_terminal,self.context.timestep)
            rewardfortwopin = rewardfortwopin + reward   #todo ??
          if self.replay.can_sample():  
            self.update_model()
        self.replay.point_to_next_episode()  #**replay got to next episode
        self.result.update_pin_result(rewardfortwopin,self.env.route)
      self.result.end_episode(self,logger,episode,results,twoPinNum)
      if early_stop == True:  #pre-training mode 
        if self.result.PosTwoPinNum/len(self.env.twopin_combo) > 0.9:
          print("early stopping when training to prevent overfitting")
          break   
    U.save_ckpt_fn(self,ckpt_path,save_ckpt)  
    success, results, solution = U.make_solutions(self,twoPinNum,netSort,twoPinNumEachNet,results)
    return results,	 solution, self.result.PosTwoPinNum, success





