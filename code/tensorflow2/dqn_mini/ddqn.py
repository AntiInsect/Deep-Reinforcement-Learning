import random
import numpy as np
import tensorflow as tf

from collections import deque
import keras
from tensorflow.keras import backend as K

from dqn import DQNAgent
from constant import DDQN_EPISODES


class DDQNAgent(DQNAgent):
    """
    A mini DDQN Agent
    """
    def __init__(self):
        super(DDQNAgent, self).__init__()
        self.target_model = None

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        # return the elements, either from `x` or `y`, depending on the `condition`
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        ret = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])

        ret.compile(
            loss=self._huber_loss,
            optimizer=keras.optimizers.Adam(lr=self.learning_rate)
        )
        return ret

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def agent_replay(self, batch_size_param):

        if len(agent.replay_buffer) <=\
                batch_size_param:
            return

        minibatch = random.sample(self.replay_buffer, batch_size_param)

        for state_param, action_param, reward_param, next_state_param, done_param in minibatch:
            target = self.model.predict(state_param)

            # crucial part of DDQN
            if done_param:
                target[0][action_param] = reward_param
            else:
                # NOTICE DETAIL
                # the next_state is predict by the "model" but updated use "target model"
                t = self.target_model.predict(next_state_param)[0]
                target[0][action_param] = reward_param + self.gamma * np.amax(t)

            self.model.fit(state_param, target, epochs=1, verbose=0)

        self.agent_lower_exploration()

    def simulation_init(self,
                        env_name_param,
                        simulation_episodes_param,
                        replay_buffer_param=deque(maxlen=2000)):
        # NOTICE DETAIL
        # do NOT pass "self" to the inheritated method
        # which will iteratively call the method
        super(DDQNAgent, self).simulation_init(env_name_param, simulation_episodes_param)

        self.target_model = self._build_model()
        self.update_target_model()

    def simulate(self):

        for e in range(self.simulation_episodes):
            state = self.simulation_env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                # self.simulation_env.render()

                state, is_done = self.simulate_step(state)

                if is_done:
                    # NOTICE DETAIL
                    # update the target model weights every time
                    # a training process is done
                    self.update_target_model()
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, DDQN_EPISODES, time, agent.epsilon))
                    break

                self.agent_replay(self.start_replay_batch_size)


if __name__ == "__main__":

    agent = DDQNAgent()
    agent.simulation_init('CartPole-v1', DDQN_EPISODES)
    agent.simulate()