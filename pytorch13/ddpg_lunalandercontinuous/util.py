import numpy as np


# according to the original paper, we need a external noise
class OUActionNoise(object):
    def __init__(self, mu, sigma=.15, theta=.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        """
        >>> noise = UOActionNoise()
        >>> noise() # everytime you call this function
                    # an inner function __call__ will
                    # be called
        """
        # the basic Gaussian Noise
        self.x = self.x + self.theta * (self.mu - self.mu) * self.dt + \
                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        return self.x

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# the ReplayBuffer class for enforcing exploration
# NOTICE that here we choose the following implementation
# for better manipulation of the type of each element
# separately
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, input_shape))
        self.next_state_mem = np.zeros((self.mem_size, input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminated_mem = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        # the mem_cntr can grow to infinity so we need to control the index
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.next_state_mem[index] = next_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminated_mem[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        terminateds = self.terminated_mem[batch]

        return states, actions, rewards, next_states, terminateds

