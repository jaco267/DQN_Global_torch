import torch as tc
from torch import nn
import torch.nn.functional as F
from Trainer.algos.noisy import NoisyLinear


class QNetwork(nn.Module):
  def __init__(self,obs_size,action_size,hid_layer=1,start_end_dim=32,emb_dim=64):
    super().__init__()
    layers = []
    layers += [nn.Linear(obs_size,start_end_dim), nn.ReLU(),
               nn.Linear(start_end_dim,emb_dim),nn.ReLU(),]
    for _ in range(hid_layer-1):
       layers+=[nn.Linear(emb_dim,emb_dim), nn.ReLU()]
    layers += [nn.Linear(emb_dim,start_end_dim), nn.ReLU(),
               nn.Linear(start_end_dim,action_size)]
    self.feature_layer = nn.Sequential(
      *layers
    )
    print(layers)
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    x = x.reshape([batch_size,-1])
    x = self.feature_layer(x)
    return x
class Noisy_Qnet(nn.Module):
  def __init__(self,obs_size,action_size,hid_layer=1,start_end_dim=32,emb_dim=64):
    """Initialization."""
    super().__init__()
    layers = []
    layers += [nn.Linear(obs_size,start_end_dim), nn.ReLU(),
               nn.Linear(start_end_dim,emb_dim),nn.ReLU(),]    
    for _ in range(hid_layer-1):
       layers+=[nn.Linear(emb_dim,emb_dim), nn.ReLU()]
    self.feature_layer = nn.Sequential(
      *layers
    )
    self.noisy_layer1 = NoisyLinear(emb_dim,start_end_dim)
    self.noisy_layer2 = NoisyLinear(start_end_dim,action_size)
  def forward(self, x: tc.Tensor) -> tc.Tensor:
    batch_size = x.shape[0]   #(bs,12,6)
    x = x.reshape([batch_size,-1])
    x = self.feature_layer(x)
    x = F.relu(self.noisy_layer1(x))
    out = self.noisy_layer2(x)
    return out
  def reset_noise(self):
    """Reset all noisy layers."""
    self.noisy_layer1.reset_noise()
    self.noisy_layer2.reset_noise()  

class Duel_QNetwork(nn.Module):
  # duel is bad 
  def __init__(self,obs_size,action_size,hid_layer=1,start_end_dim=32,emb_dim=64):
    super().__init__()
    layers = []
    layers += [nn.Linear(obs_size,start_end_dim), nn.ReLU(),
               nn.Linear(start_end_dim,emb_dim),nn.ReLU(),]
    for _ in range(hid_layer-1):
       layers+=[nn.Linear(emb_dim,emb_dim), nn.ReLU()]
    self.feature_layer = nn.Sequential(
      *layers
    )
    
    self.advantage_layer = nn.Sequential(
       nn.Linear(emb_dim,start_end_dim),nn.ReLU(),
       nn.Linear(start_end_dim,action_size)
    )
    self.value_Layer = nn.Sequential(
       nn.Linear(emb_dim,start_end_dim),nn.ReLU(),
       nn.Linear(start_end_dim,1)
    )
    print(layers,"duel adv value")
  def forward(self,x):
    batch_size = x.shape[0]   #(bs,12,6)
    x = x.reshape([batch_size,-1])
    x = self.feature_layer(x)

    value = self.value_Layer(x)
    advantage = self.advantage_layer(x)
    q = value + advantage - advantage.mean(dim=-1, keepdim=True)
    return q
  
  
def dqn_nextQ(dqn, dqn_target, b_obs_next, b_reward, gamma,b_done):
  with tc.no_grad():
    q_batch_next, _ = dqn_target(b_obs_next).max(dim=1,keepdim=True) #(32,1)
    q_batch_next = b_reward+ gamma*q_batch_next*(1-b_done)
  return q_batch_next   #32,1
def ddqn_nextQ(dqn, dqn_target, b_obs_next, b_reward, gamma,b_done):
  with tc.no_grad(): 
    selected_action = dqn(b_obs_next).argmax(dim=1,keepdim=True)  #(32,1)
    q_batch_next = dqn_target(b_obs_next).gather(dim=1,index=selected_action) #(32,1)
    q_batch_next = b_reward+ gamma*q_batch_next*(1-b_done)
  return q_batch_next   #32,1