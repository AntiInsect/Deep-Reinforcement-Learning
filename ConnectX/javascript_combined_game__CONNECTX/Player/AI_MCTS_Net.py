import numpy as np
import copy

from Game.Board import Board
import Game.Board as BOARD
from AI.MonteCarloTreeSearch import MonteCarloTreeSearch
from AI.MonteCarloTreeNode import TreeNode
from Player.Player import Player

import time


class AI_MCTS_Net(MonteCarloTreeSearch, Player):

    def __init__(self, policy_value_function, board_to_xlabel, name="AI_MCTS_Net",
                 is_training=False, search_times=2000, greedy_value=5.0,
                 is_output_analysis=True, is_output_running=True):
        """
        AI_MCTS_Net is a Player that implements MonteCarloTreeSearch,
         and makes policy of probability of action through the policy_value_function.
        :param policy_value_function: Policy value function.
        :param board_to_xlabel: Function that converts the current board to the neural network input board.
        :param name: The name of this AI.
        :param is_training:  Whether it is training mode.
        :param search_times: Monte Carlo Tree Search times.
        :param is_output_analysis: Whether to output AI analysis.
        :param is_output_running: Whether to output 'running'.
        """
        super().__init__()
        self.name = name

        self.policy_value_function = policy_value_function
        self.board_to_xlabel = board_to_xlabel
        self.is_training = is_training
        self.search_times = search_times  #  The search times of tree.
        self.greedy_value = greedy_value
        self.is_output_analysis = is_output_analysis  #  Whether to output AI analysis.
        self.is_output_running = is_output_running  #  Whether to output 'running'.

    def __str__(self):
        return "----- AI with neural network -----\n" \
               "search times: {}\n" \
               "greedy value: {}\n".format(self.search_times, self.greedy_value)

    def reset(self):
        self.root = TreeNode(prior_prob=1.0)

    def take_action(self, board: Board, is_output_action=True, running_output_function=None, is_stop=None):
        """
        The AI player take action next step.
        :param board: Current board.
        :param is_output_action:  Whether to output execution actions.
        :param running_output_function:  running output function.
        :param is_stop: Ask whether to stop.
        :return: <tuple (i, j)>  Coordinate of the action.
        """
        if is_output_action:
            print("It's turn to {0}, AI player.".format(self.name))
            print("Thinking ASAP ...")

        self.reset()
        self.run(board, self.search_times, running_output_function, is_stop=is_stop)

        # Get actions and probabilities.
        actions, probs = self.get_action_probs()

        # action -> flatten_action
        flatten_actions = []
        for one_action in actions:
            flatten_actions.append(one_action[0] * BOARD.board_size + one_action[1])

        if self.is_training:
            # add Dirichlet Noise for exploration in training.
            flatten_action = np.random.choice(flatten_actions,
                                              p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
        else:
            flatten_action = np.random.choice(flatten_actions, p=probs)

        # flatten_action -> action
        action = (flatten_action // BOARD.board_size, flatten_action % BOARD.board_size)

        board.step(action)

        if self.is_output_analysis:
            action_probs = np.zeros((BOARD.board_size, BOARD.board_size))
            # probs -> action_probs
            for one_action, one_prob in zip(actions, probs):
                action_probs[one_action[0], one_action[1]] = one_prob

            self.output_analysis(action_probs)

        if is_output_action:
            print("AI player {0} moves ({1}, {2})".format(self.name, action[0], action[1]))

        return action

    def self_play(self, temp=1e-3):
        """
        Self-play, return to all boards after the game,
        the probability of losing all positions,
        and reward of victory or lose.
        :param temp: Temperature parameter (Degree of exploration).
        :return: [(boards, all_action_probs, values)]
        """
        board_inputs, all_action_probs, current_player = [], [], []
        board = Board()
        self.reset()

        while True:
            self.run(board, self.search_times)

            # Get actions and probabilities.
            actions, probs = self.get_action_probs(temp=temp)
            action_probs = np.zeros((BOARD.board_size, BOARD.board_size))

            # actions, probs -> action_probs
            for action, prob in zip(actions, probs):
                action_probs[action[0], action[1]] = prob

            # Collect self play data.
            board_inputs.append(self.board_to_xlabel(board))
            all_action_probs.append(action_probs)
            current_player.append(board.current_player)

            # action -> flatten_action
            flatten_actions = []
            for one_action in actions:
                flatten_actions.append(one_action[0] * BOARD.board_size + one_action[1])

            # Add Dirichlet Noise for exploration in training.
            flatten_action = np.random.choice(flatten_actions,
                                              p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))

            # flatten_action -> action
            action = (flatten_action // BOARD.board_size, flatten_action % BOARD.board_size)

            board.step(action)

            # Reset the root node.
            if action in self.root.children:
                self.root = self.root.children[action]
                self.root.parent = None
            else:
                self.reset()

            is_over, winner = board.result()
            if is_over:
                values = np.zeros(len(current_player))
                if winner != 0:
                    values[np.array(current_player) == winner] = 1
                    values[np.array(current_player) != winner] = -1
                return board_inputs, all_action_probs, values

    def get_action_probs(self, temp=1e-3):
        """
        Get all the actions and probabilities.
        :param temp: Temperature parameter (Degree of exploration).
        :return: (action<tuple>, probabilities[])
        """
        action_visits = [(action, node.visited_times) for action, node in self.root.children.items()]
        actions, visits = zip(*action_visits)

        def softmax(x):
            y = np.exp(x - np.max(x))
            y /= np.sum(y)
            return y

        probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return actions, probs

    def output_analysis(self, action_probs):
        """
        Analysis of winning ratio of output pieces.
        :param action_probs:  Moving actions and probabilities.
        """
        print("----------------------------\n"
              "AI analysis:\n"
              "Format: [Decision probability(%) | #calculations]")
        # Build a 16 * 16 array.
        print_array = [["" for _ in range(BOARD.board_size + 1)] for _ in range(BOARD.board_size + 1)]

        index_string = [""] + [str(i) for i in range(BOARD.board_size)]

        # Row & Column Index.
        print_array[0] = index_string
        for row in range(BOARD.board_size):
            print_array[row + 1][0] = str(row)

        # Fill Content.
        for i in range(BOARD.board_size):
            for j in range(BOARD.board_size):
                if (i, j) in self.root.children:
                    visited_times = float(self.root.children[(i, j)].visited_times)
                    print_array[i + 1][j + 1] = "{0:.1f}%".format(action_probs[i][j] * 100) + "|" + str(
                        int(visited_times))

        # Print.
        for i in range(BOARD.board_size + 1):
            for j in range(BOARD.board_size + 1):
                print("{:<15}".format(print_array[i][j]), end="")
            print("")
        print("----------------------------")

    def run(self, start_board: Board, times, running_output_function=None, is_stop=None):
        """
        Monte Carlo Tree, start searching...
        :param start_board: The start state of the board.
        :param times: run times.
        :param running_output_function: running output function.
        :param is_stop: Ask whether to stop.
        """
        for i in range(times):
            board = copy.deepcopy(start_board)
            if is_stop is not None:
                if is_stop():
                    running_output_function("Game stopped.")
                    return
            if i % 50 == 0 and running_output_function is not None:
                running_output_function("{} / {}".format(i, times))
                time.sleep(0.01)
            if i % 20 == 0 and self.is_output_running:
                print("\rrunning: {} / {}".format(i, times), end="")

            # Expand node.
            node, value = self.traverse(self.root, board)

            # Backpropagate.
            node.backpropagate(-value)
        print("\r                      ", end="\r")

    def traverse(self, node: TreeNode, board: Board):
        """
        Expand node.
        :param node: Current node.
        :param board: The board.
        :return: (<TreeNode>, value<int>) value。
        Expanded nodes, and the value to be backpropagated.
        """
        while True:
            if len(node.children) == 0:
                break
            action, node = node.choose_best_child(c=self.greedy_value)
            board.step(action)

        # game over?
        is_over, winner = board.result()
        if is_over:
            if winner == board.current_player:
                value = 1.0
            elif winner == -board.current_player:
                value = -1.0
            else:
                value = 0.0
            return node, value

        # Use the strategy value function to decide the current action probability and evaluate the value.
        action_probs, value = self.policy_value_function(board)

        for action, probability in action_probs:
            _ = node.expand(action, probability)

        return node, value
