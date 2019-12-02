import gym
from tqdm import tqdm

from agent import Agent


NUM_EPISODE = 1000

# create new env
env = gym.make('LunarLanderContinuous-v2')
# generate agent
agent = Agent(alpha=0.000025, beta=0.00025, n_states=8, tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)
# store the reward of each episode (tracjectory)
scores = []

for _ in tqdm(range(NUM_EPISODE)):
    # actually we should use "observation"
    # get the initial state
    state = env.reset()
    done = False
    score = 0

    while not done:
        # get the action using actor network
        action = agent.act(state)
        # environment step to get next state and reward
        new_state, reward, done, _ = env.step(action)
        # put the transition into the replay buffer
        agent.remember(state, action, reward, new_state, int(done))

        # train the agent
        agent.step()
        # store the reward
        score += reward
        # change to next transition
        state = new_state

    scores.append(score)
