import copy

import torch as T 
import torch.nn.functional as F

from utils import OUNoise, ReplayBuffer
from network import Critic, Actor


# the DDPG agent
class Agent(object):
    def __init__(self, env, alpha, beta, tau, gamma,
                 state_dim = 8, action_dim = 2, max_replay_size = 1000000,
                 l1_dim = 400, l2_dim = 300, batch_size=64):

        self.env = env
        self.max_action = float(env.action_space.high[0])

        self.alpha = alpha # learning rate for actor network
        self.beta = beta # learning rate for critic network
        self.tau = tau # polyak averaging parameter
        self.gamma = gamma # discount factor of reward

        self.update_actor_count = 0
        self.update_actor_freq  = 2 

        self.policy_noise = .2
        self.noise_clip = .5

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.batch_size = batch_size
        self.max_replay_size = max_replay_size

        # build the agent
        self.build_agent()
        # with "tau = 1", we initialize the target network the same as the main network
        self.update_target_network(tau = 1)

    def build_agent(self):
        # build the actor-critic network and also their target networks
        self.actor = Actor(self.state_dim, self.action_dim, self.l1_dim, self.l2_dim,self.alpha)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(self.state_dim, self.action_dim, self.l1_dim, self.l2_dim,self.beta)
        self.target_critic = copy.deepcopy(self.critic)

        # build the replaybuffer
        self.replaybuffer = ReplayBuffer(self.max_replay_size, self.state_dim, self.action_dim)
        # build the OUNoise for action selection 
        self.noise = OUNoise(self.action_dim)

    def act(self, state):
        state = T.tensor(state, dtype=T.float)
        action = self.actor(state)
        noisy_action = action + T.tensor(self.noise(), dtype=T.float)
        return noisy_action.cpu().detach().numpy()

    # store transition into the replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.replaybuffer.store(state, action, reward, next_state, done)

    def sample_replaybuffer(self):
        # sample from the ReplayBuffer
        states, actions, rewards, next_states, dones = self.replaybuffer.sample(self.batch_size)
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards, dtype=T.float)
        next_states = T.tensor(next_states, dtype=T.float)
        dones = T.tensor(dones)

        return states, actions, rewards, next_states, dones

    def step(self):
        # we cannot learn before the amount of transitions inside
        # the replay buffer is larger than the batch size
        if self.replaybuffer.mem_cntr < self.batch_size:
            return
        
        self.update_actor_count += 1
            
        # get transition samples from replayer buffer
        states, actions, rewards, next_states, dones = self.sample_replaybuffer()
        # update the critic network
        self.update_critic(states, actions, rewards, next_states, dones)

        if self.update_actor_count % self.update_actor_freq == 0:
            # update the actor network
            self.update_actor(states)
            # update target network parameters
            self.update_target_network()
        
    def update_critic(self, states, actions, rewards, next_states, dones):
        with T.no_grad():
            # Select action according to policy and add clipped noise
            noise = (T.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value and use the minimum of them
            target_Q1, target_Q2 = self.target_critic(next_states, next_action)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = [rewards[j] + self.gamma * target_Q[j] * dones[j] for j in range(self.batch_size)]
            # reshape the variable
            target_Q = T.tensor(target_Q)
            target_Q = target_Q.view(self.batch_size, 1)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def update_actor(self, states):
        # here we use the output from the actor network NOT the noisy action
        # because we only need to enforce exploration in the when actual interactions
        # happen in the environment
        actions = self.actor(states)
        actor_loss = - self.critic.q1_forward(states, actions).mean()
        
        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def update_target_network(self, tau=None):
        tau = self.tau if tau is None else tau

        # polyak averaging to update the target critic network
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # polyak averaging to update the target actor network
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
