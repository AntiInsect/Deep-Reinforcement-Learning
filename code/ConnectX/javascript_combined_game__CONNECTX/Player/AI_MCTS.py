import numpy as np
import copy
import random

from Game.Board import Board
import Game.Board as BOARD
from AI.MonteCarloTreeSearch import MonteCarloTreeSearch
from AI.MonteCarloTreeNode import TreeNode
from Player.Player import Player

import time


class AI_MCTS(MonteCarloTreeSearch, Player):

    def __init__(self, name="AI_MCTS", search_times=2000, greedy_value=5.0,
                 is_output_analysis=True, is_output_running=True):
        super().__init__()
        self.name = name

        self.search_times = search_times  # The search times of tree.
        self.greedy_value = greedy_value  # The greedy value.
        self.is_output_analysis = is_output_analysis  # Whether to output AI analysis.
        self.is_output_running = is_output_running  # 'running'。 Whether to output 'running'.

    def __str__(self):
        return "----- AI with pure MCTS -----\n" \
               "search times: {}\n" \
               "greedy value: {}\n".format(self.search_times, self.greedy_value)

    def reset(self):
        self.root = TreeNode(prior_prob=1.0)

    def take_action(self, board: Board, is_output_action=True, running_output_function=None, is_stop=None):
        """
        The AI player take action next step.
        :param board: Current board.
        :param is_output_action: Whether to output action information.
        :param running_output_function: running output function.
        :param is_stop: Ask whether to stop.
        :return: <tuple (i, j)> Coordinate of the action.
        """
        if is_output_action:
            print("It's turn to {0}, AI player.".format(self.name))
            print("Thinking ASAP ...")

        self.reset()
        self.run(board, self.search_times, running_output_function, is_stop=is_stop)
        action, _ = self.root.choose_best_child(0)
        board.step(action)

        if self.is_output_analysis:
            self.output_analysis()

        if is_output_action:
            print("AI player {0} moves ({1}, {2})".format(self.name, action[0], action[1]))

        return action

    def output_analysis(self):
        """
        Analysis of winning ratio of output pieces.
        """
        print("----------------------------\n"
              "AI analysis:\n"
              "Format: [odds(%) | #calculations]")
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
                    reward = float(self.root.children[(i, j)].reward)
                    print_array[i + 1][j + 1] = "{0:.1f}%".format(reward / visited_times * 100) + "|" + str(
                        int(visited_times)) if visited_times != 0 else 0

        # 输出。 Print.
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

            # 扩展节点。
            node = self.traverse(self.root, board)
            node_player = board.current_player

            winner = self.rollout(board)

            value = 0
            if winner == node_player:
                value = 1
            elif winner == -node_player:
                value = -1

            node.backpropagate(-value)
        print("\r                      ", end="\r")

    def traverse(self, node: TreeNode, board: Board):
        """
        Expand node.
        :param node: Current node.
        :param board: The board.
        :return: <TreeNode> Expanded nodes.
        """
        while True:
            if len(node.children) == 0:
                break
            action, node = node.choose_best_child(c=self.greedy_value)
            board.step(action)

        is_over, _ = board.result()
        if is_over:
            return node

        # Expand all child node.
        actions = board.available_actions
        probs = np.ones(len(actions)) / len(actions)

        for action, prob in zip(actions, probs):
            _ = node.expand(action, prob)

        return node

    def rollout(self, board: Board):
        """
        Simulation.
        :param board: The board.
        :return: winner<int> winner.
        """
        while True:
            is_over, winner = board.result()
            if is_over:
                break
            # Decision making next step.
            self.rollout_policy(board)
        return winner

    def rollout_policy(self, board: Board):
        """
        Decision function, random decision here.
        :param board: The board.
        """
        # Randomly execute actions.
        board.step(random.choice(list(board.available_actions)))
