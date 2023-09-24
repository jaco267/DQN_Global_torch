import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
import Trainer.algos.resultUtils  as U

np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
class HiddenLayer(nn.Module):
   def __init__(self, emb_dim):
      super().__init__()
      self.nn1 = nn.Linear(emb_dim,emb_dim)
      self.ffn = nn.Sequential(
          nn.Linear(emb_dim, 4 * emb_dim), nn.ReLU(),
          nn.Linear(4 * emb_dim, emb_dim),
      )
   def forward(self,x):
      out1 = self.nn1(x)
      x = x+F.relu(out1)
      ffn = self.ffn(x)
      x = x+F.relu(ffn)
      return x

class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size,hid_layer=3,start_end_dim=32,emb_dim=64):
    super().__init__()
    self.obs_embedding = nn.Linear(obs_size,emb_dim)
    self.hidden_layers = nn.Sequential(
       *[
          HiddenLayer(emb_dim)
          for _ in range(hid_layer)
       ]
    )
    if hid_layer == 0:
       #*** original implementation only used 3 layers  
       #https://github.com/haiguanl/DQN_GlobalRouting/blob/master/GlobalRoutingRL/DQN_Implementation.py
       self.hidden_layers = nn.Sequential(
             nn.Linear(emb_dim, emb_dim),nn.ReLU(),
             nn.Linear(emb_dim, emb_dim),nn.ReLU(),
       )
    self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),nn.ReLU(),
            nn.Linear(emb_dim, action_size),
        )
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    x = x.reshape([batch_size,-1])
    x = self.obs_embedding(x)
    x = self.hidden_layers(x)
    x = self.ffn(x)
    return x
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
class DQN_Agent(): 
    def __init__(self, gridgraph,hid_layer=1,emb_dim=64,self_play_episode_num=150,context_len=5):
        # as well as training parameters - number of episodes / iterations, etc.
      self.env = gridgraph
      self.action_size = self.env.action_size  #6
      obs_size = self.env.obs_size        #12

      self.dqn = QNetwork(obs_size,self.action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
      self.dqn_target = QNetwork(obs_size,self.action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
      self.dqn_target.load_state_dict(self.dqn.state_dict())
      self.dqn_target.eval()
      #** replay buffer
      self.replay = ReplayBuffer(obs_dim=obs_size)
      #** explore
      self.epsilon = 0.05
      #** training
      self.gamma = 0.95
      self.batch_size = 32
      self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
      
      self.num_train_steps = 0
      self.target_update_frequency = 100
      self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
      self.criterion = nn.MSELoss()
      #** training results   (used in resultUtils.py)
      self.result = U.Result() 
    @tc.no_grad()
    def get_action(self,obs):
      rnd = np.random.rand()
      if rnd <= self.epsilon:
        return np.random.randint(self.action_size)
      else:
        q_values = self.dqn(tc.FloatTensor(obs).to(device)) 
        return tc.argmax(q_values).item()
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
        b_action =tc.LongTensor(samples["acts"].reshape(-1, 1)).to(device)  # action
        b_reward = tc.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)  # reward
        b_obs_next = tc.FloatTensor(samples["next_obs"]).to(device)
        b_done = tc.FloatTensor(samples["done"].reshape(-1, 1)).to(device) # is term
        
        q_batch = self.dqn(b_obs)#.gather(1,b_action)   
        # action generate by q_batch
        with tc.no_grad():  #* trainable false
            q_batch_next = self.dqn_target(b_obs_next).max(dim=1,keepdim=True)[0].detach()
            y_batch = (b_reward+self.gamma*q_batch_next*(1-b_done))#.to(device) 
            targetQ = q_batch.cpu()
            # print(targetQ.shape,b_action.shape,y_batch.shape,"sss")
            targetQ[tc.arange(self.batch_size),b_action.cpu().reshape(-1)] = y_batch.cpu().reshape(-1)
        loss = self.criterion(q_batch,targetQ.to(device))
        return loss
    def train(self,twoPinNumEachNet,  
        netSort:list, #--netsort [0,7,17,15,4,3, ...,8,18] len 20---
        ckpt_path,
        logger, save_ckpt=False, load_ckpt= True,
        early_stop=False,
    ):                  #* the training/testing curve will be saved as a .npz file in ../data/
        if load_ckpt:                              #, 
            U.load_ckpt_or_pass(self,ckpt_path)
        results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
        twoPinNum = len(self.env.twopin_combo)  
        for episode in range(self.max_episodes):	
            self.result.init_episode(agent=self)
            for pin in range(twoPinNum):
                state = self.env.reset(pin)   
                is_terminal = False;   rewardfortwopin = 0
                while not is_terminal:
                  obs = self.env.state2obsv()
                  with tc.no_grad():
                      action = self.get_action(obs)
                      nextstate, reward, is_terminal, _ = self.env.step(action)  #* agent step
                      self.result.update_episode_reward(reward)   
                      obs_next = self.env.state2obsv()
                      self.replay.store(obs, action, reward, obs_next, is_terminal)
                      rewardfortwopin = rewardfortwopin + reward   
                  if self.replay.can_sample():
                    self.update_model()
                self.result.update_pin_result(rewardfortwopin,self.env.route)
            self.result.end_episode(self,logger,episode,results,twoPinNum)
            if early_stop == True:  #pre-training mode 
                if self.result.PosTwoPinNum/len(self.env.twopin_combo) > 0.9:
                    print("early stopping when training to prevent overfitting")
                    break   
        U.save_ckpt_fn(self,ckpt_path,save_ckpt)  
        success, results, solution = U.make_solutions(self, twoPinNum,netSort,twoPinNumEachNet,results)
			
        return results,	 solution, self.result.PosTwoPinNum, success





