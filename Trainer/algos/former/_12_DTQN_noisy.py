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
import math
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet. """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super().__init__()
        self.in_features = in_features  #input size of linear module
        self.out_features = out_features #output size of linear module
        self.std_init = std_init #initial std value
        # mean value weight parameter
        self.weight_mu = nn.Parameter(tc.Tensor(out_features, in_features))
        # std value weight parameter
        self.weight_sigma = nn.Parameter(tc.Tensor(out_features, in_features))

        # weight_epsilon is params that don't need to be trained
        self.register_buffer("weight_epsilon", tc.Tensor(out_features, in_features))
        # mean value bias parameter
        self.bias_mu = nn.Parameter(tc.Tensor(out_features))
        # std value bias parameter
        self.bias_sigma = nn.Parameter(tc.Tensor(out_features))
        # bias_epsilon is params that don't need to be trained
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
            requires_grad=False
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
  def __init__(self,obs_size,action_size,embed_dim = 64,context_len=5,num_heads=8,hid_layers=3): #* 128 is a little too large...
    super().__init__()
    print("=========context_len...",context_len,"========!!!!!!")
    # lay = [32,64,32]
    self.position_embedding = nn.Parameter(tc.zeros(1,context_len,embed_dim),requires_grad=True)
    self.obs_embedding = nn.Linear(obs_size, embed_dim)
    self.dropout = nn.Dropout(0) #todo
    self.transformer_layers = nn.Sequential(
        *[  TransformerLayer(num_heads=num_heads, embed_dim=embed_dim, history_len=context_len)
            for _ in range(hid_layers)
         ]
    )
    self.adv_hid_layer = NoisyLinear(embed_dim, 32)
    self.adv_lay = NoisyLinear(32,action_size)

    self.val_hid_layer = NoisyLinear(embed_dim, 32)
    self.val_lay = NoisyLinear(32,1)

    self.history_len = context_len
    self.apply(init_weights)#  Applies ``fn`` recursively to every submodule 
  def forward(self,obs):
    # x (bs,seq_len,obs_dim) 
    bs,seq_len,obs_dim = obs.shape
    obs= tc.flatten(obs,start_dim=0,end_dim=1)  #(bs*seq_len,obs_dim) = (bs*l,12)
    obs_embed = self.obs_embedding(obs)  #(bs*seq_len,outer_embed_size)
    obs_embed = obs_embed.reshape(bs,seq_len,obs_embed.size(-1)) #(bs,seq_len,outer_embed_size)
    working_memory = self.transformer_layers(obs_embed)
    
    adv_hid = F.relu(self.adv_hid_layer(working_memory))
    advantage = self.adv_lay(adv_hid)[:, -(seq_len):, :]

    val_hid = F.relu(self.val_hid_layer(working_memory))
    value = self.val_lay(val_hid)[:, -(seq_len):, :] #[32,50,6]

    q = value + advantage - advantage.mean(dim=-1, keepdim=True)
    return  q    #[32,50,6]
  def reset_noise(self):
    self.adv_hid_layer.reset_noise()
    self.val_hid_layer.reset_noise()
    self.adv_lay.reset_noise()
    self.val_lay.reset_noise()
    
class ReplayBuffer:
  """A simple numpy replay buffer."""
  def __init__(self,  obs_dim: int=12, frame_size: int=50000, batch_size: int = 32,   context_len=50,  env_max_steps =  100  #* gridword max steps
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

class Context:
    """A Dataclass dedicated to storing the agent's history (up to the previous `max_length`)"""
    def __init__( self,
        context_length: int,  # 50 maximum number of transitions to store_
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
  def __init__(self, gridgraph,hid_layer=1,emb_dim=64,self_play_episode_num=20,context_len=5):
    print(f"----DTQN_normal_noisy_agent--context_len {context_len}-")
    self.context_len = context_len 
    self.env = gridgraph
    self.action_size = self.env.action_size  #6
    obs_size = self.env.obs_size        #12
    env_max_step = self.env.max_step
    assert env_max_step==100
    self.dqn = QNetwork(obs_size,self.action_size,embed_dim=emb_dim,hid_layers=hid_layer,context_len=context_len).to(device)
    self.dqn_target = QNetwork(obs_size,self.action_size,embed_dim=emb_dim,hid_layers=hid_layer,context_len=context_len).to(device)
    self.dqn_target.load_state_dict(self.dqn.state_dict())
    self.dqn_target.eval()
    #** PER
    self.beta = 0.6   #* from 0.6 to 1 # importance sampling at the end   
    alpha = 0.2   #* temperature  (lower the alpha, closer to uniform)
    self.prior_eps = 1e-6 #* guarentees every transition can be sampled
    self.replay = ReplayBuffer(
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
    # weights = tc.FloatTensor(samples["weights"].reshape(-1,1)).to(device) 
    #todo ??  reshape(-1,1)???
    # indices = samples['indices']
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
    frames_i = 0
    num_frames = self.max_episodes*twoPinNum
    for episode in range(self.max_episodes):	
      self.result.init_episode(agent=self)
      for pin in range(twoPinNum):
        #* PER: increase beta (importance sampling)
        fraction = min(frames_i/num_frames,1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)  #beta ~ 1
        state = self.env.reset(pin)     #*  New loop!
        obs = self.env.state2obsv()
        #***   context_reset(env.reset())
        self.context.reset(obs)
        self.replay.initialize_episode_buffer(obs)   
        is_terminal = False;    rewardfortwopin = 0
        while not is_terminal:
          with tc.no_grad():
            action = self.get_action()
            nextstate, reward, is_terminal, _ = self.env.step(action)  #* agent step  
            self.result.update_episode_reward(reward)
            obs = self.env.state2obsv()
            #*update_context_and_buffer
            self.context.add_transition( obs, action, reward, is_terminal )
            self.replay.store(obs,action,reward,is_terminal,self.context.timestep)
            rewardfortwopin = rewardfortwopin + reward   #todo ??
          if self.replay.can_sample():  
            self.update_model()
        self.replay.point_to_next_episode()  #**replay got to next episode
        frames_i+=1
        self.result.update_pin_result(rewardfortwopin,self.env.route)
      self.result.end_episode(self,logger,episode,results,twoPinNum)
      if early_stop == True:  #pre-training mode 
         if self.result.PosTwoPinNum/len(self.env.twopin_combo) > 0.9:
            print("early stopping when training to prevent overfitting")
            break   
    
    if save_ckpt:
      print('\nSave model')
      tc.save(self.dqn.state_dict(),ckpt_path)
      print(f"Model saved in path: {ckpt_path}")
    else:
      print("dont save model")
    solution = self.result.best_route
    assert len(solution) == twoPinNum
    for i in range(len(netSort)):
      results['solutionDRL'].append([])
    success = 0
    if self.result.PosTwoPinNum  == twoPinNum:
      dumpPointer = 0
      for i in range(len(netSort)):
        netToDump = netSort[i]
        for j in range(twoPinNumEachNet[netToDump]):
          results['solutionDRL'][netToDump].append(solution[dumpPointer])
          dumpPointer = dumpPointer + 1
      success = 1
    else:
      results['solutionDRL'] = solution
  
    return results,	 solution,  self.result.PosTwoPinNum,success