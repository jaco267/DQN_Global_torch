import torch as tc
from torch import nn
import torch.nn.functional as F

import numpy as np

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
            x,x,x,  attn_mask=self.attn_mask[: x.size(1), : x.size(1)],  #*this will trigger warning
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