import os
import numpy as np
import copy

import torch as T 
import torch.nn as nn
import torch.nn.functional as F

from utils import OUNoise, ReplayBuffer
from network import Critic, Actor


# the DDPG agent
class Agent(object):
    def __init__(self, env, alpha, beta, tau, gamma,
                 state_dim = 8, action_dim = 2, max_replay_size = 1000000,
                 l1_dim = 400, l2_dim = 300, batch_size=64):

        self.env = env
        self.alpha = alpha # learning rate for actor network
        self.beta = beta # learning rate for critic network
        self.tau = tau # polyak averaging parameter
        self.gamma = gamma # discount factor of reward

        self.update_actor_count = 0
        self.update_actor_freq  = 2 

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.batch_size = batch_size
        self.max_replay_size = max_replay_size

        self.policy_noise = .2
        self.noise_clip = .5
        self.max_action = float(env.action_space.high[0])

        # build the agent
        self.build_agent()
        # with "tau = 1", we initialize the target network the same as the main network
        self.update_target_network(tau = 1)

    def build_agent(self):
        # build the actor-critic network and also their target networks
        self.actor = Actor(self.alpha, self.state_dim, self.l1_dim, self.l2_dim, self.action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(self.beta, self.state_dim, self.l1_dim, self.l2_dim, self.action_dim)
        self.target_critic = copy.deepcopy(self.critic)

        # build the replaybuffer
        self.replaybuffer = ReplayBuffer(self.max_replay_size, self.state_dim, self.action_dim)
        # build the OUNoise for action selection 
        self.noise = OUNoise(self.action_dim)

    def act(self, state):
        # if we only want to predict (forward), it is no need to use "train()" mode
        # "eval()" turn off the BatchNormalization and Dropout
        self.actor.eval()
        state = T.tensor(state, dtype=T.float)
        action = self.actor.forward(state)
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
            self.update_actor(states, actions)
            # update target network parameters
            self.update_target_network()
        
    def update_critic(self, states, actions, rewards, next_states, dones):
        with T.no_grad():
            # Select action according to policy and add clipped noise
            noise = (T.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_states, next_action)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = rewards + dones * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    # def update_critic(self, states, actions, rewards, next_states, dones):
    #     # update the critic network
    #     self.target_actor.eval()
    #     target_actions = self.target_actor.forward(next_states)
    #     self.target_critic.eval()
    #     target_critic_values = self.target_critic.forward(next_states, target_actions)
    #     self.critic.eval()
    #     critic_values = self.critic.forward(states, actions)

    #     target_critic_values = [rewards[j] + self.gamma * target_critic_values[j] * dones[j] for j in range(self.batch_size)]
    #     # reshape the variable
    #     target_critic_values = T.tensor(target_critic_values)
    #     target_critic_values = target_critic_values.view(self.batch_size, 1)

    #     # set the mode to "train" (turn up BatchNormalization and Dropout)
    #     self.critic.train()
    #     # In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
    #     # because PyTorch accumulates the gradients on subsequent backward passes
    #     self.critic.optimizer.zero_grad()
    #     # calculate the MSE loss 
    #     critic_loss = F.mse_loss(target_critic_values, critic_values)
    #     # backpropagation and optimize
    #     critic_loss.backward()
    #     self.critic.optimizer.step()

    def update_actor(self, states, actions):
        # NOTICE that here we want to "forward",so no need to use model.train()
        self.critic.eval()
        self.actor.optimizer.zero_grad()

        # here we use the output from the actor network NOT the noisy action
        # because we only need to enforce exploration in the when actual interactions
        # happen in the environment
        actions = self.actor.forward(states)
        self.actor.train()
        actor_loss = T.mean( - self.critic.q1_forward(states, actions))
        actor_loss.backward()
        self.actor.optimizer.step()

    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau

        # polyak averaging to update
        # update the target critic network
        critic_state_dict = dict(self.critic.named_parameters())
        target_critic_dict = dict(self.target_critic.named_parameters())
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

        # update the target actor network
        actor_state_dict = dict(self.actor.named_parameters())
        target_actor_dict = dict(self.target_actor.named_parameters())
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
