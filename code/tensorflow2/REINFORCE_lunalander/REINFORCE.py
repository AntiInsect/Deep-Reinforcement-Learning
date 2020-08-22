from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np 


class Agent(object):
    def __init__(self, ALPHA, GAMMA=.99, n_actions=4, layer1_size=16,
                layer2_size=16, input_dims=128, fname='reinforce.h5'):
        
        self.gamma = GAMMA # discount factor
        self.lr = ALPHA
        self.n_actions = n_actions
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.input_dims = input_dims
        self.model_file = fname    

        # self.action_space = np.arrange(n_actions)
        # NOTICE that the commented line using numpy seems more
        # convenient but it is not (hard to handle the types), 
        # we'd better use python list at first and then use change
        # it to the numpy array when necessary manipulation is required
        self.action_space = [i for i in range(n_actions)]
        
        self.policy, self.predict = self.build_policy_network()
        
        self.reset()

    # policy network
    def build_policy_network(self):
        # the following implmentation is a very clear buildup
        # when just use the keras.Sequentail (evrey hello world program)
        # to build, we must make sure the predict step have the right
        # input. In our case, we don't need advantages to do predictions
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            # clip out the 0 value in the case of using log likihood
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * advantages)

        policy  = Model(input=[input, advantages], output=[probs])
        predict = Model(input=[input], output=[probs])

        # Adam or SGD is fine for calculating the Gradient
        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        return policy, predict

    # observation passed into the function by the env
    def choose_action(self, observation):
        # add a fake "batch dimension" at the 0 index axis
        # it allows us to align with the Input Layer we just
        # define above
        state = observation[np.newaxis, :]

        probs = self.predict.predict(state)[0]
        # print("check out put of the predict", self.predict.predict(state))

        # no exploration action selection
        action = np.random.choice(self.action_space, p=probs)

        return action

    # NOTICE that we here are dealing with the REINFORCE algorithm
    # which use the Monte Carlo Method i.e. does not focus on the
    # one step transitions
    def store_transition(self, observation, action, reward):
        self.state_mem.append(observation)
        self.action_mem.append(action)
        self.reward_mem.append(reward)


    def learn(self):
        # Now, change the type of the memory buffer for
        # easy calculation
        state_mem = np.array(self.state_mem)
        action_mem = np.array(self.action_mem)
        reward_mem = np.array(self.reward_mem)

        # NOTICE one-hot encoding
        actions = np.zeros([len(action_mem), self.n_actions])
        actions[np.arange(len(action_mem)), action_mem] = 1

        # calculate the Monte Carlo Returm from different timesteps
        # actually, it is more easily to do this backwards
        G = np.zeros_like(reward_mem)
        for t in range(len(reward_mem)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_mem)):
                G_sum += reward_mem[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # NOTICE calculate the baseline
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.advantage = (G - mean) / std

        # pass the value (same with the input layer) to the network
        cost = self.policy.train_on_batch([state_mem, self.advantage], actions)
        self.reset()

    def reset(self):
        self.state_mem = []
        self.action_mem = []
        self.reward_mem = []

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)


