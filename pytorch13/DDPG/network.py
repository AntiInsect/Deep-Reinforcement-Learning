import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# to be extremely clear about how we should handle the pytorch NN construction
# the following implementation will separate all the components from a single
# network building process

# the critic network to learn how to estimate the "max action-value"
class Critic(nn.Module):
    def __init__(self, lr, state_dim, l1_dim, l2_dim, action_dim):

        # recive general purpose variables here, leave sepecial ones to sepecific functions
        super(Critic, self).__init__()
        
        # network parameters
        self.lr = lr
        self.state_dim = state_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.action_dim = action_dim

        # the pytorch network construction routine
        self.build_network()
        self.set_optimizer()

    # for this function, we should give more problem-specific names to
    # help us understand the processing better
    def forward(self, state, action):

        state_value = self.fc1(state)
        # we'd better to do the batch norm first before the relu
        state_value = self.bn1(state_value)
        # DO not forget the activation function
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        # according to the original paper, we include the actions here
        action_value = self.separate_action_layer(action)
        action_value = F.relu(action_value)

        # combine the state_value and action_value
        state_action_value = F.relu(T.add(state_value, action_value))
        # get the output which is the estimation of max action-value
        state_action_value = self.q(state_action_value)

        return state_action_value

    # setup the architecture and as the name goes, we have not
    # actually make any connection between layers but just scratch
    # the feature of each layer 
    def build_network(self):
        # the first fully connected layer
        self.fc1 = nn.Linear(self.state_dim, self.l1_dim)

        # "f" here means the "fan-in" of the layer in the original paper
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])

        # Initialize the layer with uniform distribution
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        # according to the original paper, we need to introduce batch normalization
        # here to avoid the effect the covariate shift of the underlying of the low
        # dimensional input
        self.bn1 = nn.LayerNorm(self.l1_dim)

        # the second fully connected layers
        self.fc2 = nn.Linear(self.l1_dim, self.l2_dim)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.l2_dim)

        # according to the original paper, we do not include the action until the
        # 2nd hidden layer, so we need a separate layer here
        # NOTICE the input and output dims here 
        self.separate_action_layer = nn.Linear(self.action_dim, self.l2_dim)

        # according to the original paper, we add this extra layer to
        # ensure the initial outputs for the policy and value estimates were near zero.
        f3 = .003

        # NOTICE that here is "l2_dim" not the "action_dim"
        # we name the final layer "Q" to indicate the esimation of Q-value 
        self.q = nn.Linear(self.l2_dim, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

    # set network optimizer, Adam is the to-go choice
    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


# the Actor network 
class Actor(nn.Module):
    def __init__(self, lr, state_dim, l1_dim, l2_dim, action_dim):
        super(Actor, self).__init__()
        self.lr = lr
        self.state_dim = state_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.action_dim = action_dim

        self.buleprint_network()
        self.set_optimizer()

    def forward(self, state):
        action = self.fc1(state)
        action = self.bn1(action)
        action = F.relu(action)

        action = self.fc2(action)
        action = self.bn2(action)
        action = F.relu(action)
        
        # according to the original paper, we use the "tanh" at the final
        # layer to bound the actions
        action = T.tanh(self.mu(action))
        return action

    def buleprint_network(self):
        self.fc1 = nn.Linear(self.state_dim, self.l1_dim)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.l1_dim)

        self.fc2 = nn.Linear(self.l1_dim, self.l2_dim)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.l2_dim)

        f3 = .003
        self.mu = nn.Linear(self.l2_dim, self.action_dim)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
