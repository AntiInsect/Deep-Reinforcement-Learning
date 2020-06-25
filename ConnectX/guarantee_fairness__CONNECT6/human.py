# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
@modifier: Junguang Jiang
"""

from __future__ import print_function
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class Human(object):
    """
    human player
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


if __name__ == '__main__':
    import sys, getopt

    height = 10
    width = 10
    n_in_row = 6
    use_gpu = False
    n_playout = 800
    model_file = "model/10_10_6_best_policy_3.model"
    ai_first=True

    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    best_policy = PolicyValueNet(width, height, model_file=model_file, use_gpu=use_gpu)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=n_playout)
    human = Human()
    game.start_play(human, mcts_player, start_player=ai_first, is_shown=1)
