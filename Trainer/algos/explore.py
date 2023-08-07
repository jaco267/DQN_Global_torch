import torch as tc
import numpy as np
device = 'cuda' if tc.cuda.is_available() else 'cpu'
def get_eps_action(agent,obs):
  with tc.no_grad():
    rnd = np.random.rand()
    if rnd <= agent.epsilon:
      return np.random.randint(agent.action_size)
    else:
      q_values = agent.dqn(tc.FloatTensor(obs).to(device)) 
      return tc.argmax(q_values).item()
def get_noisy_action(agent,obs):
  with tc.no_grad():
    q_values = agent.dqn(tc.FloatTensor(obs).to(device)) 
    return tc.argmax(q_values).item()