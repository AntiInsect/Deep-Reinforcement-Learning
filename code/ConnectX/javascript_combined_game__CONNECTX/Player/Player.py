from abc import ABCMeta, abstractmethod
from Game.Board import Board


class Player(metaclass=ABCMeta):

    @abstractmethod
    def take_action(self, board: Board, is_output_action=True, running_output_function=None, is_stop=None):
        """
        What the player should do next.
        :param board: Current board.
        :param is_output_action: Whether to output action information.
        :param running_output_function: running output function.
        :param is_stop: Ask whether to stop.
        :return: <tuple (i, j)> Coordinate of the action.
        """
