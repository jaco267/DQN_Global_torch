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