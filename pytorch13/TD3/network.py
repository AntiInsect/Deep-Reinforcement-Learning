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
        self.action_dim = action_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim

        # the pytorch network construction routine
        self.build_network()
        self.set_optimizer()

    # for this function, we should give more problem-specific names to
    # help us understand the processing better
    def forward(self, state, action):
        return self.q1_forward(state, action), self.q2_forward(state, action)

    def q1_forward(self, state, action):
        s1 = F.relu(self.bn1(self.fc1(state)))
        s1 = self.bn2(self.fc2(s1))
        # according to the original paper, we include the actions here
        a1 = F.relu(self.separate_action_layer(action))
        # combine the state_value and action_value
        q1 = F.relu(T.add(s1, a1))
        q1 = self.q(q1)
        
        return q1

    def q2_forward(self, state, action):
        s2 = F.relu(self.bn3(self.fc3(state)))
        s2 = self.bn4(self.fc4(s2))
        # according to the original paper, we include the actions here
        a2 = F.relu(self.separate_action_layer2(action))
        # combine the state_value and action_value
        q2 = F.relu(T.add(s2, a2))
        q2 = self.q2(q2)
        
        return q2

    def build_network(self):
        self.build_network1()
        self.build_network2()

    # setup the architecture and as the name goes, we have not
    # actually make any connection between layers but just scratch
    # the feature of each layer 
    def build_network1(self):
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

    def build_network2(self):
        self.fc3 = nn.Linear(self.state_dim, self.l1_dim)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.l1_dim)
        self.fc4 = nn.Linear(self.l1_dim, self.l2_dim)
        f4 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(self.l2_dim)

        self.separate_action_layer2 = nn.Linear(self.action_dim, self.l2_dim)
        f5 = .003
        self.q2 = nn.Linear(self.l2_dim, 1)
        T.nn.init.uniform_(self.q2.weight.data, -f5, f5)
        T.nn.init.uniform_(self.q2.bias.data, -f5, f5)

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

        self.build_network()
        self.set_optimizer()

    def forward(self, state):
        action = F.relu(self.bn1(self.fc1(state)))
        action = F.relu(self.bn2(self.fc2(action)))
        
        # according to the original paper, we use the "tanh" at the final
        # layer to bound the actions
        action = T.tanh(self.mu(action))
        return action

    def build_network(self):
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
