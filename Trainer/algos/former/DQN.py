
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
class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size):
    super().__init__()
    lay = [32,64,32]
    self.nn1 = nn.Linear(obs_size,lay[0])
    self.nn2 = nn.Linear(lay[0],lay[1])
    self.nn3 = nn.Linear(lay[1],lay[2])
    self.nn4 = nn.Linear(lay[2],action_size)
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    
    x = x.reshape([batch_size,-1])
    x = F.relu(self.nn1(x))
    x = F.relu(self.nn2(x))
    x = F.relu(self.nn3(x))
    x = self.nn4(x)
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
    def __init__(self, gridgraph,self_play_episode_num=20):
      print("----DQN_agent---")
        # as well as training parameters - number of episodes / iterations, etc.
      self.env = gridgraph
      action_size = self.env.action_size  #6
      obs_size = self.env.obs_size        #12

      self.epsilon = 0.05
      self.gamma = 0.95
      self.max_episodes = self_play_episode_num    #20 #200#10000 #20000
      self.batch_size = 32
      self.dqn = QNetwork(obs_size,action_size).to(device)
      self.dqn_target = QNetwork(obs_size,action_size).to(device)
      self.dqn_target.load_state_dict(self.dqn.state_dict())
      self.dqn_target.eval()

      self.replay = ReplayBuffer(obs_dim=obs_size)
		
      self.optim = tc.optim.Adam(self.dqn.parameters(),lr=0.0001)
    def epsilon_greedy_policy(self, q_values):
      # Creating epsilon greedy probabilities to sample from.
      rnd = np.random.rand()
      if rnd <= self.epsilon:
        return np.random.randint(len(q_values))
      else:
        return np.argmax(q_values)
    def train(self,
	   	twoPinNumEachNet,  # len = netNum in one file = 20 , value = netPinNum - 1  ex. [3, 2, 2, 1, 4, 2, 3,..., 4]
	   	netSort:list,  # ---netsort [0, 7, 17, 15, 4, 3, 1, ...,8, 18] len 20---
	   	savepath,  #"../model_(train/test)"   #model will be saved to ../model/
		model_file=None  # if model_file = None, training; if given, testing  #* if testing using training function, comment burn_in in Router.py
    ):                  #* the training/testing curve will be saved as a .npz file in ../data/
        """
          train our network. 
          # If training without experience replay_memory, then you will interact with the environment 
          # in this function, while also updating your network parameters. 
          
          # If you are using a replay memory, you should interact with environment here, and store these 
          # transitions to memory, while also updating your model.
        """
        if os.path.exists(f"{savepath}model.ckpt"):  #load chpt
            print(f"loading {savepath}model.ckpt")
            statedict = tc.load(f"{savepath}model.ckpt")   
            self.dqn.load_state_dict(statedict)
        results = {	'solutionDRL':[], 'reward_plot_combo': [],	'reward_plot_combo_pure': [],}
        '''
		#*  self.gridgraph.twopin_combo [      
		[(2,6,1, 28, 62), (1,7,1, 11, 77)]    correspond to	twoPinNumEachNet   3 
		[(2,6,1, 28, 62), (3,1,1, 30, 17)] ,
		[(2,6,1, 28, 62),(7,7,1, 78, 75)] 
	
		[(6, 2, 1, 65, 22), (0, 3, 1, 1, 39)] correspond to	twoPinNumEachNet   2
	    [(0, 3, 1, 1, 39), (3, 6, 1, 31, 68)]		
		  ... 
		] connect
		len(self.gridgraph.twopin_combo)  == 49    net num ==20
		'''
        criterion = nn.MSELoss()
		
		
        twoPinNum = len(self.env.twopin_combo)  #* ex. 49,  20net has 49 pin connect, avg_connect/net ~=2.5
        update_count = 0
        for episode in range(self.max_episodes):	
			# print("epi",episode)
            for pin in range(twoPinNum):
                # print(f"\repisode  {episode}",end="")
                state, reward_plot, is_best = self.env.reset(self.max_episodes)     #*  New loop!
                reward_plot_pure = reward_plot-self.env.posTwoPinNum*100
                if (episode) % twoPinNum == 0:      #*  after one episode
                    results['reward_plot_combo'].append(reward_plot)
                    results['reward_plot_combo_pure'].append(reward_plot_pure)
                is_terminal = False;
				
				#*copy q_net's param  to t_net every 100 episode

                rewardfortwopin = 0
                while not is_terminal:
                  
                  obs = self.env.state2obsv()
                  with tc.no_grad():
                      q_values = self.dqn(tc.from_numpy(obs).to(tc.float32).to(device))  #*  action shape
  						# print(q_values,"111")
                      action = self.epsilon_greedy_policy(q_values.cpu().numpy())
                      nextstate, reward, is_terminal, debug = self.env.step(action)  #* agent step  
                      obs_next = self.env.state2obsv()
                      self.replay.store(obs, action, reward, obs_next, is_terminal)   #* save to replay buffer
  						
                      rewardfortwopin = rewardfortwopin + reward   #todo ??
                  if len(self.replay) >= self.batch_size:
                    update_count+=1
                    samples = self.replay.sample_batch()       
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
                    loss = criterion(q_batch,targetQ.to(device))
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    if update_count % 100 == 0:
                       self.dqn_target.load_state_dict(self.dqn.state_dict())
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





