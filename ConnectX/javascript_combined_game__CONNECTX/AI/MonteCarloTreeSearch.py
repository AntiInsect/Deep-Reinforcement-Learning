from abc import ABCMeta, abstractmethod

from Game.Board import Board


class MonteCarloTreeSearch(metaclass=ABCMeta):

    def __init__(self):
        self.root = None

    @abstractmethod
    def run(self, board: Board, times):
        """
        Monte Carlo Tree, start searching...
        :param board:
        :param times: run times.
        :return: the best action.
        """
