import random
import numpy as np

from dqn import DQNAgent
from constant import DQN_BATCH_EPISODES


class DQNBatchAgent(DQNAgent):
    """
    A mini DQN_Batch Agent
    """

    def agent_replay(self, batch_size_param):
        # check if we can start the replay operation
        if len(self.replay_buffer) <= \
                self.start_replay_batch_size:
            return

        minibatch = random.sample(self.replay_buffer, batch_size_param)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])

        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]

        self.agent_lower_exploration()
        return loss

    def simulate(self):

        for e in range(self.simulation_episodes):
            state = self.simulation_env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(500):
                # env.render()

                state, is_done = self.simulate_step(state)

                if is_done:
                    print(
                        "episode: {}/{}, score: {}, error: {:.2}"
                        .format(e, self.simulation_episodes, time, self.epsilon)
                    )
                    break

                loss = self.agent_replay(self.start_replay_batch_size)
                if loss is not None and time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, DQN_BATCH_EPISODES, time, loss))


if __name__ == "__main__":

    agent = DQNBatchAgent()
    agent.simulation_init('CartPole-v1', DQN_BATCH_EPISODES)
    agent.simulate()
