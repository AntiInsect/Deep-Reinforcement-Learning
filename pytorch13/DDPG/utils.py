import numpy as np


# according to the original paper, to gain better performance on exploration
# we need to inject a external noise into the off-policy algorithm. Specificall,
# we add the OUNoise to the actor network output to get a random action.
class OUNoise(object):
    def __init__(self, size, mu=.0, sigma=.15, theta=.2, dt=1e-2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.zeros_like(self.mu)

    def __call__(self):
        self.state += self.theta * (self.mu - self.state) * self.dt + \
                      self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        return self.state


# the ReplayBuffer class for enforcing exploration and reduce correlation between
# transition. This is a common trick for Deep Q-learning implmentations along with
# the inner target network structure
# NOTICE that here we choose the following implementation for better manipulation
# of the type of each element separately
class ReplayBuffer(object):
    def __init__(self, max_size, n_states, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, n_states))
        self.next_state_mem = np.zeros((self.mem_size, n_states))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminated_mem = np.zeros(self.mem_size, dtype=np.int32)
        
    def store(self, state, action, reward, next_state, done):
        # the mem_cntr can grow to infinity so we need to control the index
        index = self.mem_cntr % self.mem_size
        self.state_mem[index] = state
        self.next_state_mem[index] = next_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminated_mem[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample(self, batch_size):
        # first decide the sample range
        mem_range = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(mem_range, batch_size)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        terminateds = self.terminated_mem[batch]

        return states, actions, rewards, next_states, terminateds
