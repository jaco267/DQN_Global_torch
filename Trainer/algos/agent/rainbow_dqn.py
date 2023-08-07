

import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
import Trainer.algos.resultUtils  as U
from Trainer.algos.replay_buffer import ReplayBuffer,PER,NstepBuffer 
from Trainer.algos.Q_net import QNetwork,Duel_QNetwork,Noisy_Qnet,\
                                  dqn_nextQ,ddqn_nextQ
from Trainer.algos.explore import get_eps_action,get_noisy_action

np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
class RainMode():
  def __init__(self,rainbow_mode:dict):
    self.double:bool = rainbow_mode['double']
    self.duel:bool   = rainbow_mode['duel']
    self.noisy:bool = rainbow_mode['noisy']
    self.per:bool = rainbow_mode['per']
    self.cat:bool = rainbow_mode['cat']
    self.nstep:bool = rainbow_mode['nstep']
  def nextQ(self):
    if self.double == True:
      print("double_dqn nextQ")
      return ddqn_nextQ
    else:
      print("dqn nextQ")
      return dqn_nextQ
  @property
  def Qnet(self):
    if self.duel == True and self.noisy == True:
        raise Exception('error... not implement duel && noisy') 
    elif self.duel == True and self.noisy == False:
      print("dueling Q net")   
      return Duel_QNetwork
    elif self.duel == False and self.noisy == True:
      print("noisy net")
      return Noisy_Qnet
    elif self.duel == False and self.noisy == False:
      print("normal Q net")
      return QNetwork
    else:
        raise Exception('error... not implement error')
  def get_action_fn(self):
    if self.noisy == True:
      print("get some noisy action")
      return get_noisy_action
    else:
      print("get epsilon greedy action")
      return get_eps_action
class DQN_Agent(): 
  def __init__(self, env, rainbow_mode:dict, hid_layer=1,
               emb_dim=64,self_play_num=150,context_len=5):
    print("----DQN_agent_general---")
      # as well as training parameters - number of episodes / iterations, etc.
    self.r = RainMode(rainbow_mode)
    self.env = env
    self.action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    #** agent
    self.dqn = self.r.Qnet(obs_size,self.action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
    self.dqn_target = self.r.Qnet(obs_size,self.action_size,hid_layer=hid_layer,emb_dim=emb_dim).to(device)
    #** training
    self.gamma = 0.95
    self.batch_size = 32
    self.max_episodes = self_play_num    #20 #200#10000 #20000
    
    self.num_train_steps = 0
    self.target_update_frequency = 100
    self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
    self.criterion = nn.MSELoss()
    
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()
    #** dqn|ddqn next q
    self.nextQ = self.r.nextQ()
    #** replay buffer
    if self.r.per:
      alpha = 0.2
      self.beta = 0.6
      self.prior_eps = 1e-6
      print("---PER---")
      self.replay = PER(obs_dim=obs_size,size=50000,batch_size=self.batch_size,alpha=alpha)
    else:
      self.replay = ReplayBuffer(obs_dim=obs_size,size=50000,batch_size=self.batch_size)
    if self.r.nstep:
      print("nstep....")
      # memory for 1-step Learning
      self.replay = NstepBuffer(obs_dim=obs_size,size=50000,batch_size=self.batch_size,n_step=1)
      self.n_step = 3
      self.replay_n =  NstepBuffer(obs_dim=obs_size,size=50000,batch_size=self.batch_size,n_step=self.n_step,gamma=self.gamma)
    #** explore
    if  self.r.noisy == False:
      self.epsilon = 0.05   #* used in get_eps_action
    self.get_action = self.r.get_action_fn()
    
    #** training results   (used in resultUtils.py)
    self.result = U.Result() 
  def update_model(self,):
    if self.r.per == True:
      samples = self.replay.sample_batch(self.beta)   
      weights = tc.FloatTensor(samples["weights"].reshape(-1, 1)).to(device)
      indices = samples["indices"]
      elementwise_loss = self._compute_dqn_loss(samples,self.gamma)
      loss = tc.mean(elementwise_loss * weights)
    elif self.r.nstep == True:  #todo merge Nstep buffer with normal buffer
      samples = self.replay.sample_batch()
      indices = samples["indices"]
      loss = self._compute_dqn_loss(samples, self.gamma)
    else:
      samples = self.replay.sample_batch()   
      loss = self._compute_dqn_loss(samples,self.gamma)
    if self.r.nstep == True:
      samples = self.replay_n.sample_batch_from_idxs(indices)
      gamma = self.gamma ** self.n_step
      n_loss = self._compute_dqn_loss(samples,gamma)
      loss += n_loss    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    if self.r.per == True:
      loss_for_prior = elementwise_loss.detach().cpu().numpy()
      new_priorities = loss_for_prior + self.prior_eps
      self.replay.update_priorities(indices, new_priorities)
    self.num_train_steps+=1
    if self.r.noisy == True:
      self.dqn.reset_noise()
      self.dqn_target.reset_noise()
    if self.num_train_steps % self.target_update_frequency == 0:
        self.dqn_target.load_state_dict(self.dqn.state_dict())
  def _compute_dqn_loss(self,samples: Dict[str,np.ndarray],gamma=0.95)->tc.Tensor:
    b_obs = tc.FloatTensor(samples["obs"]).to(device)
    b_action =tc.LongTensor(samples["acts"].reshape(-1, 1)).to(device)  # action
    b_reward = tc.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)  # reward
    b_obs_next = tc.FloatTensor(samples["next_obs"]).to(device)
    b_done = tc.FloatTensor(samples["done"].reshape(-1, 1)).to(device) # is term

    q_batch = self.dqn(b_obs).gather(1,b_action)      #(32,1)
    q_batch_next = self.nextQ(self.dqn,self.dqn_target,b_obs_next,b_reward,gamma,b_done)
    if self.r.per == True:
      #PER calculate element-wise dqn loss
      elementwise_loss = F.smooth_l1_loss(q_batch, q_batch_next, reduction="none")
      return elementwise_loss
    else:
      loss = self.criterion(q_batch,q_batch_next)
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
    pin_count = 0;  #*for PER
    for episode in range(self.max_episodes):	
      self.result.init_episode(agent=self)
      for pin in range(twoPinNum):
        state = self.env.reset(pin)   
        is_terminal = False;   rewardfortwopin = 0
        if self.r.per == True:
          fraction = min(pin_count / twoPinNum*self.max_episodes, 1.0)
          self.beta = self.beta + fraction * (1.0 - self.beta)
        while not is_terminal:
          obs = self.env.state2obsv()
          with tc.no_grad():
            action = self.get_action(self,obs)
            nextstate, reward, is_terminal, _ = self.env.step(action)  #* agent step
            self.result.update_episode_reward(reward)   
            obs_next = self.env.state2obsv()
            if self.r.nstep:
              one_step_transition = self.replay_n.store(obs, action, reward, obs_next, is_terminal)
              if one_step_transition:
                self.replay.store(*one_step_transition)
            else:
              self.replay.store(obs, action, reward, obs_next, is_terminal)
            rewardfortwopin = rewardfortwopin + reward   
          if self.replay.can_sample():
            self.update_model()
        self.result.update_pin_result(rewardfortwopin,self.env.route)
        pin_count+=1
      self.result.end_episode(self,logger,episode,results,twoPinNum)
      if early_stop == True:  #pre-training mode 
        if self.result.PosTwoPinNum/len(self.env.twopin_combo) > 0.9:
          print("early stopping when training to prevent overfitting")
          break   
    U.save_ckpt_fn(self,ckpt_path,save_ckpt)  
    success, results, solution = U.make_solutions(self, twoPinNum,netSort,twoPinNumEachNet,results)
			
    return results,	 solution, self.result.PosTwoPinNum, success





