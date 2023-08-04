import numpy as np
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