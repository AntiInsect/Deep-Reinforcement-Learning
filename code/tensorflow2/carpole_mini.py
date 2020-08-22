#!/usr/bin/env python3
# modified version from 
# https://medium.com/coinmonks/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python-50a4d4296687

import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class TrainingEnv(object):
    _env_name = 'CartPole-v1'

    @classmethod
    def print_env_name(cls):
        print("the training environment is : ", cls._env_name)

    def __init__(self, _goal_steps=500, _score_requirement=60,
                 _inital_games=10000, _game_iter=100):
        self.__goal_steps = _goal_steps
        self.__score_requirement = _score_requirement
        self.__inital_games = _inital_games
        self.__game_iter = _game_iter
        self.__env = gym.make(self._env_name)
        self.__env.reset()

    # make the env "printable"
    def __str__(self):
        return "standard steps for each game : " + str(self.__goal_steps) + "\n" + \
               "score requirement for each game : " + str(self.__score_requirement) + "\n" +\
               "dataset init iterations : " + str(self.__inital_games) + "\n" + \
               "training game bot iterations : " + str(self.__game_iter) + "\n"

    def set_env(self, _goal_steps=500, _score_requirement=60,
                _inital_games=10000, _game_iter=100):
        self.__goal_steps = _goal_steps
        self.__score_requirement = _score_requirement
        self.__inital_games = _inital_games
        self.__game_iter = _game_iter

    def get_goal_steps(self):
        return self.__goal_steps

    def get_score_requirement(self):
        return self.__score_requirement

    def get_inital_games(self):
        return self.__inital_games

    def get_game_iter(self):
        return self.__game_iter

    # init the _training_data
    # the data come from the env but used by the model
    def create_data(self):
        training_data = []
        accepted_scores = []
        for game_index in range(self.__inital_games):
            score = 0
            game_memory = []
            previous_observation = []

            for step_index in range(self.__goal_steps):
                action = random.randrange(0, 2)
                observation, reward, done, info = self.__env.step(action)
                if len(previous_observation) > 0:
                    game_memory.append([previous_observation, action])
                previous_observation = observation
                score += reward
                if done:
                    break

            if score >= self.__score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    output = []
                    if data[1] == 1:
                        output = [0, 1]
                    elif data[1] == 0:
                        output = [1, 0]
                    training_data.append([data[0], output])

            self.__env.reset()
        return training_data


class TrainingModel(TrainingEnv):
    def __init__(self, _training_env=TrainingEnv(),
                 _goal_steps=500, _score_requirement=60,
                 _inital_games=10000, _game_iter=100, _epochs=10):

        _training_env.set_env(_goal_steps, _score_requirement, _inital_games, _game_iter)
        super(TrainingModel, self).__init__(_training_env.get_goal_steps(),
                                            _training_env.get_score_requirement(),
                                            _training_env.get_inital_games(),
                                            _training_env.get_game_iter())

        self._traning_data = self.create_data()

        x = np.array([i[0] for i in self._traning_data]).reshape(-1, len(self._traning_data[0][0]))
        y = np.array([i[1] for i in self._traning_data]).reshape(-1, len(self._traning_data[0][1]))

        self.__input_size = len(x[0])
        self.__output_size = len(y[0])
        self.__epochs = _epochs

        self.__model = Sequential()
        self.__model.add(Dense(128, input_dim=self.__input_size, activation='relu'))
        self.__model.add(Dense(52, activation='relu'))
        self.__model.add(Dense(self.__output_size, activation='linear'))
        self.__model.compile(loss='mse', optimizer=Adam())
        self.__model.fit(x, y, epochs=self.__epochs)

    def __str__(self):
        return "the input size is : " + str(self.__input_size) + "\n" + \
               "the output size is : " + str(self.__output_size) + "\n" + \
               "the training episode is : " + str(self.__epochs) + "\n"

    def set_epochs(self, _epoches):
        self.__epochs = _epoches


class CarPole(TrainingModel):
    def __init__(self, _training_env=TrainingEnv(),
                 _goal_steps=500, _score_requirement=60,
                 _inital_games=10000, _game_iter=100, _epochs=10):

        super(CarPole, self).__init__(_training_env, _goal_steps, _score_requirement,
                                      _inital_games, _game_iter, _epochs)

    def carpole_run(self, if_render=False):
        scores = []
        choices = []
        for each_game in range(self.__game_iter):
            score = 0
            prev_obs = []
            for step_index in range(self.__goal_steps):
                if if_render:
                    self.__env.render()
                if len(prev_obs) == 0:
                    action = random.randrange(0, 2)
                else:
                    action = np.argmax(self.__model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])

                choices.append(action)
                new_observation, reward, done, info = self.__env.step(action)
                prev_obs = new_observation
                score += reward
                if done:
                    break

            self.__env.reset()
            scores.append(score)


# run the carpole agent
if __name__ == "__main__":
    mycarpole = CarPole()
    mycarpole.carpole_run()






