import math
import torch as tc
from torch import nn
import torch.nn.functional as F
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