import numpy as np
import torch as tc
from torch import nn
import torch.nn.functional as F
import random
import math
np.random.seed(10701)
random.seed(10701)
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
from Trainer.algos.transformer_utils.transformer import TransformerLayer,init_weights
from Trainer.algos.noisy import NoisyLinear
class DTQN_noisy_net(nn.Module):
  def __init__(self,obs_size,action_size,embed_dim = 64,
               context_len=5,hid_layers=3,num_heads=8,): #* 128 is a little too large...
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