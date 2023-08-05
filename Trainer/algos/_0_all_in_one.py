import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple
from Trainer.algos.segment_tree import MinSegmentTree, SumSegmentTree

import Trainer.algos.resultUtils  as U  #import end_episode
from torch.nn.utils import clip_grad_norm_
np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
from net_utils import DTQN_noisy_net
from replay_buffer import Episode_ReplayBuffer
from data_context import Context

class DQN_Agent(): 
  def __init__(self, gridgraph,hid_layer=1,emb_dim=64,self_play_episode_num=20,context_len=5):
    print(f"----DTQN_normal_noisy_agent--context_len {context_len}-")
    self.context_len = context_len 
    self.env = gridgraph
    self.action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    env_max_step = self.env.max_step  #100
    self.dqn = DTQN_noisy_net(obs_size,self.action_size,embed_dim=emb_dim,
                        context_len=context_len,hid_layers=hid_layer,).to(device)
    self.dqn_target = DTQN_noisy_net(obs_size,self.action_size,embed_dim=emb_dim,
                        context_len=context_len,hid_layers=hid_layer,).to(device)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()
    #** PER
    self.beta = 0.6   #* from 0.6 to 1 # importance sampling at the end   
    alpha = 0.2   #* temperature  (lower the alpha, closer to uniform)
    self.prior_eps = 1e-6 #* guarentees every transition can be sampled
    self.replay = Episode_ReplayBuffer(
        obs_dim=obs_size,context_len=context_len,env_max_steps=env_max_step)
    # NoisyNet: All attributes related to epsilon are removed
    self.gamma = 0.95
    self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
    self.batch_size = 32
    
    self.grad_norm_clip = 1.0
    self.num_train_steps = 0
    self.target_update_frequency = 100 #update target network every 100 steps
    self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
    self.context = Context(context_len,self.action_size,obs_size)
    self.criterion = nn.MSELoss(reduction="none")
    #** training results   (used in resultUtils.py)
    self.result = U.Result() 
  @tc.no_grad()
  def get_action(self, ):
    # NoisyNet: no epsilon greedy action selection
    context_obs_tensor = tc.FloatTensor( #50
          self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
      ).unsqueeze(0).to(device)   #todo investigate the shape  so  seq_len range from 1 to 50,  means output of transformer also range from 1 to 50 ??
    q_values = self.dqn(context_obs_tensor)
    return tc.argmax(q_values[:, -1, :]).item()   #argmax(1,-1,action_dim)  #todo  the shape
  def update_model(self,):
    samples = self.replay.sample_batch()       
    elementwise_loss = self._compute_dqn_loss(samples)
    loss = tc.mean(elementwise_loss )  #importance sampling
    self.optim.zero_grad()
    loss.backward()
    clip_grad_norm_(self.dqn.parameters(), 10.0)
    self.optim.step()
    # PER: update priorities
    loss_for_prior = elementwise_loss.detach().cpu().numpy()
    new_priorities = loss_for_prior + self.prior_eps  
    #* PER larger the loss, larger the priorities
    # self.replay.update_priorities(indices, new_priorities)
    self.num_train_steps+=1

    # NoisyNet: reset noise
    self.dqn.reset_noise()
    self.dqn_target.reset_noise()
    #
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
      q_batch_next = self.dqn_target(b_obs_next).gather(#ddqn # [batch-size x hist-len x n-actions] 
         2, self.dqn(b_obs_next).argmax(dim=2).unsqueeze(-1)
      ).squeeze()
      # print(b_reward.shape,q_batch_next.shape,b_done.shape)
      targets = b_reward.squeeze()+self.gamma*q_batch_next*(1-b_done.squeeze())
    q_values = q_values[:, -self.context_len :]
    targets = targets[:, -self.context_len :]  
    elementwise_loss = tc.mean(self.criterion(q_values,targets),dim=-1) #*no mean at batch
    return elementwise_loss
  def train(self,
    twoPinNumEachNet,  # len = netNum in one file = 20 , value = netPinNum - 1  ex. [3, 2, 2, 1, 4, 2, 3,..., 4]
    netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
    ckpt_path,  #"../model_(train/test)"   #model will be saved to ../model/
    logger,
    save_ckpt=False, # if model_file = None, training; if given, testing 
    load_ckpt= True,
    early_stop=False,
  ):           
    if load_ckpt:
      U.load_ckpt_or_pass(self,ckpt_path)
    results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
    twoPinNum = len(self.env.twopin_combo)
    for episode in range(self.max_episodes):	
      self.result.init_episode(agent=self)
      for pin in range(twoPinNum):                     #,
        state = self.env.reset(pin)     #,
        obs = self.env.state2obsv()
        self.context.reset(obs) #context_reset(env.reset())
        self.replay.initialize_episode_buffer(obs)   
        is_terminal = False;    rewardfortwopin = 0
        while not is_terminal:
          with tc.no_grad():
            action = self.get_action()
            nextstate, reward, is_terminal, _ = self.env.step(action)  # agent step  
            self.result.update_episode_reward(reward)  #,
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

    return results,	 solution,  self.result.PosTwoPinNum,success