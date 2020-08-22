import gym
import numpy as np
import random
import math
from itertools import count
import matplotlib.pyplot as plt

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


env = gym.make("BipedalWalker-v2")


# Create observation sample
observation_sample = []
for _ in range(100):
    obs = env.reset()
    observation_sample.append(obs)
    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        observation_sample.append(obs)
    

class LinearApprox(object):
    def __init__(self, num_params, lr=1e-3):
        self.weights = np.zeros((num_params, 1))
        self.alpha = lr
        
    def featurize(self):
        raise NotImplementedError
    
    def cal(self, _input):
        return _input.T.dot(self.weights)
    
    def update_weights(self, delta_weights):
        self.weights += delta_weights

class Actor(LinearApprox):
    def __init__(self, num_params, observation_examples, gamma_list, lr=1e-3):
        super().__init__(num_params, lr=lr)
        # define & normalize scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        # define featurizer
        num_featurizers = 4  # featurizers per 1 action dimension
        assert num_params % num_featurizers == 0
        assert len(gamma_list) == num_featurizers * NUM_ACTIONS
        num_components = num_params // num_featurizers  # components per 1 action dimension

        self.featurizer = sklearn.pipeline.FeatureUnion([
                        (str(i), RBFSampler(gamma=gamma_list[i], n_components=num_components))
                        for i in range(num_featurizers * NUM_ACTIONS)])

        self.featurizer.fit(self.scaler.transform(observation_examples))
    
    def featurize(self, state):
        scaled = self.scaler.transform(state.reshape(1, -1))
        return self.featurizer.transform(scaled)[0].reshape(NUM_ACTIONS, self.weights.shape[0]).T
    
    def cal(self, state):
        return self.featurize(state).T.dot(self.weights)
    
    def update_weights(self, state, critic_weights):
        self.weights += self.alpha * self.grad(state).dot(self.grad(state).T.dot(critic_weights)) 
    
    def grad(self, state):
        return self.featurize(state)

class Critic(LinearApprox):
    def __init__(self, num_params, lr=1e-3):
        super().__init__(num_params, lr=lr)
        
    def featurize(self, state, action, _actor):
        return _actor.grad(state).dot(action)
    
    def cal(self, state, action, _actor, _V):
        return (action - _actor.cal(state)).T.dot(_actor.grad(state).T).dot(self.weights) + _V.cal(state)
    
    def update_weights(self, td_error, state, action, new_state, _actor, _U):
        td_correction = GAMMA * self.featurize(new_state, _actor.cal(new_state), _actor) * \
                                self.featurize(state, action, _actor).T.dot(_U.weights)
        self.weights += self.alpha * td_error * self.featurize(state, action, _actor) - self.alpha * td_correction

class StateValueFunc(LinearApprox):
    def __init__(self, num_params, observation_examples, gamma_list, lr=1e-3):
        super().__init__(num_params, lr=lr)
        # define & normalize scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        # define featurizer
        num_featurizers = 4  # featurizers per 1 action dimension
        assert num_params % num_featurizers == 0
        assert len(gamma_list) == num_featurizers 
        num_components = num_params // num_featurizers  # components per 1 action dimension

        self.featurizer = sklearn.pipeline.FeatureUnion([
                        (str(i), RBFSampler(gamma=gamma_list[i], n_components=num_components))
                        for i in range(num_featurizers)])

        self.featurizer.fit(self.scaler.transform(observation_examples))
    
    def featurize(self, state):
        scaled = self.scaler.transform(state.reshape(1, -1))
        return self.featurizer.transform(scaled)[0].reshape(-1, 1)
    
    def cal(self, state):
        return self.featurize(state).T.dot(self.weights)
    
    def update_weights(self, td_error, state, action, new_state, _critic, _actor, _U):
        td_correction = GAMMA * self.featurize(new_state) * \
                                _critic.featurize(state, action, _actor).T.dot(_U.weights)
        self.weights += self.alpha * td_error * self.featurize(state) - self.alpha * td_correction

class TDC(object):
    def __init__(self, num_params, lr=1e-3):
        self.weights = np.zeros((num_params,1))
        self.alpha = lr
    
    def update_weights(self, td_error, state, action, _critic, _actor):
        self.weights += self.alpha * (td_error - _critic.featurize(state, action, _actor).T.dot(self.weights)) * \
                                        _critic.featurize(state, action, _actor)


NUM_ACTIONS = 4
NUM_PARAMS = 4 * 15
GAMMA = 0.99

# statistic
avg_reward_per_episode = []
num_step_per_episode = []
avg_l1_V_weights_per_episode = []

def behave_policy(_input):
    return np.array([np.random.normal(_in) for _in in _input]).reshape(-1, 1)

behave_policy(np.array([0, 1, 2]))


a_gamma_list = [5., 2., 1., 0.5, 5., 2., 1., 0.5, 5., 2., 1., 0.5, 5., 2., 1., 0.5]
# actor = Actor(NUM_PARAMS, observation_sample, a_gamma_list)
# critic = Critic(NUM_PARAMS)
# V = StateValueFunc(NUM_PARAMS, observation_sample, [5., 2., 1., 0.5]) 
# U = TDC(NUM_PARAMS)


max_episode = 31

for i_episode in range(max_episode):
    reward_list = []
    l1_V_weights = []
    state = env.reset()
    for step in count():
        action = behave_policy(actor.cal(state))
        new_state, reward, done, _ = env.step(action)
        
        # calculate td error
        if not done:
            td_target = reward + GAMMA * critic.cal(new_state, actor.cal(new_state), actor, V)
        else:
            td_target = reward
        
        td_error = td_target - critic.cal(state, action, actor, V)
        
        # update weights
        actor.update_weights(state, critic.weights)
        critic.update_weights(td_error, state, action, new_state, actor, U)
        V.update_weights(td_error, state, action, new_state, critic, actor, U)
        U.update_weights(td_error, state, action, critic, actor)
        
        reward_list.append(reward)
        l1_V_weights.append(np.absolute(V.weights).sum())
        if done:
            print("Episode %d finish after %d step due to %.1f" % (i_episode, step + 1, reward))
            print("----------------------------------------")
            break
        else:
            if step % 200 == 0: print("Episode %d\t TD error: %.4f" % (i_episode, td_error))
        
        # Move on
        state = new_state

    # calculate statistic
    avg_reward_per_episode.append(np.mean(reward_list))
    num_step_per_episode.append(step)
    avg_l1_V_weights_per_episode.append(np.mean(l1_V_weights))


plt.figure(1)
plt.subplot(311)
plt.plot(avg_reward_per_episode)
plt.subplot(312)
plt.plot(num_step_per_episode)
plt.subplot(313)
plt.plot(avg_l1_V_weights_per_episode)
plt.show()

def checkout_actor(_actor):
    state = env.reset()
    ans = 0
    for step in count():
        act = _actor.cal(state)
        new_state, reward, done, _= env.step(act)
        env.render()
        if done:
            print("Episode %d [ACTOR] Finish after %d step" % (i_episode, step+1))
            print("-------------------------------------")
            ans = step + 1
            break
        # Move on 
        state = new_state
    return ans

checkout_actor(actor)

env.close()
