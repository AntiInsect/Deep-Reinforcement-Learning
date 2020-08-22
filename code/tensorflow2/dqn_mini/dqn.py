import random
import gym
# from gym.wrappers import Monitor
import numpy as np

from collections import deque
# use tensorflow.keras will hugely slow down the process
import keras

from constant import DQN_EPISODES


class DQNAgent:
    """
    A mini DQN Agent
    """
    def __init__(self,
                 gamma_param=0.95,
                 epsilon_param=1.0,
                 epsilon_min_param=0.01,
                 epsilon_decay_param=0.995,
                 learning_rate_param=0.001,
                 start_replay_batch_size_param=32):

        self.gamma = gamma_param
        self.epsilon = epsilon_param
        self.epsilon_min = epsilon_min_param
        self.epsilon_decay = epsilon_decay_param
        self.learning_rate = learning_rate_param
        self.start_replay_batch_size = start_replay_batch_size_param

        # properities uninitialized until simulation
        self.simulation_env = None
        self.simulation_episodes = None
        self.replay_buffer = None
        self.state_size = None
        self.action_size = None
        self.model = None

    # the method in class begins with "_" is a protected method
    def model_build(self):

        ret = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])

        ret.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=self.learning_rate)
        )

        return ret

    def model_load(self, name_param):

        self.model.load_weights(name_param)

    def model_save(self, name_param):

        self.model.save_weights(name_param)

    def agent_store_experience(self,
                       state_param,
                       action_param,
                       reward_param,
                       next_state_param,
                       done_param):

        self.replay_buffer.append((
            state_param,
            action_param,
            reward_param,
            next_state_param,
            done_param
        ))

    def agent_act(self, state_param):

        # epsilon exploration case
        # NOTICE DETAIL
        # by sampling a random number from a uniform distribution over [0, 1)
        # and compare with the "epsilon"
        if np.random.rand() <= self.epsilon:
            # random.randint(a, b) == randrange(a, b+1)
            return random.randrange(self.action_size)

        # normal greedy case
        # make a predict on a predict which is the Q-Learning Algriithm

        # NOTICE DETAIL
        # the result of model.predict is in form of [[...]]
        # since we can pass more inputs as once
        # so we need to first "[0]" and then find the max index
        act_values = self.model.predict(state_param)

        return np.argmax(act_values[0])

    # decaying exploration
    # when the experiment goes on, we do not need a lot of
    # exploration as the beginning situation
    def agent_lower_exploration(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def agent_replay(self, batch_size_param):

        # check if we can start the replay operation
        if len(self.replay_buffer) <= \
                self.start_replay_batch_size:
            return

        # memory replay techique
        minibatch = random.sample(self.replay_buffer, batch_size_param)

        # handle each "input"
        for state_param, action_param, reward_param, next_state_param, done_param in minibatch:
            # do not forget to handle the special case that
            # the state is the terminate state
            q_target = reward_param
            if not done_param:
                q_target = reward_param + \
                           self.gamma * np.amax(self.model.predict(next_state_param)[0])

            target_f = self.model.predict(state_param)
            target_f[0][action_param] = q_target

            # the default epoches is 1, so removed

            # NOTICE DETAIL
            # Keras does all the work of subtracting the
            # target from the neural network output and squaring it
            self.model.fit(state_param, target_f, verbose=0)

        self.agent_lower_exploration()

    def simulation_init(self,
                        env_name_param,
                        simulation_episodes_param,
                        replay_buffer_param=deque(maxlen=2000)):

        self.simulation_env = gym.make(env_name_param)
        # self.simulation_env = Monitor(self.simulation_env, './video')
        self.simulation_episodes = simulation_episodes_param
        self.replay_buffer = replay_buffer_param
        self.state_size = self.simulation_env.observation_space.shape[0]
        self.action_size = self.simulation_env.action_space.n
        self.model = self.model_build()

    def simulate_step(self, curr_state_param):

        action = self.agent_act(curr_state_param)
        next_state, reward, done, _ = self.simulation_env.step(action)

        # handle special case
        if done:
            reward = -10

        # NOTICE DETAIL
        # the same as the above act_values
        # the state is also in form of [[...]]
        next_state = np.reshape(next_state, [1, self.state_size])

        # store the experience into the replay buffer
        self.agent_store_experience(
            curr_state_param,
            action,
            reward,
            next_state,
            done
        )

        return next_state, done

    def simulate(self):

        for e in range(self.simulation_episodes):
            state = self.simulation_env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                # self.simulation_env.render()

                state, is_done = self.simulate_step(state)

                if is_done:
                    print(
                        "episode: {}/{}, score: {}, error: {:.2}"
                        .format(e, self.simulation_episodes, time, self.epsilon)
                    )
                    break

                self.agent_replay(self.start_replay_batch_size)


if __name__ == "__main__":

    agent = DQNAgent()
    agent.simulation_init('CartPole-v1', DQN_EPISODES)
    agent.simulate()



