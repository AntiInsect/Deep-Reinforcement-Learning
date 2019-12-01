import gym
import numpy as np

from tqdm import tqdm

from REINFORCE import Agent

if __name__ == '__main__':
    agent = Agent(ALPHA=0.0005, input_dims=8, GAMMA=0.99,
                  n_actions=4, layer1_size=64, layer2_size=64)

    env = gym.make('LunarLander-v2')
    score_history = []

    num_episodes = 2000

    for i in tqdm(range(num_episodes)):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        agent.learn()
        print('episode: ', i,'score: %.1f' % score, 'average score %.1f' % np.mean(score_history[max(0, i-100):(i+1)]))
        