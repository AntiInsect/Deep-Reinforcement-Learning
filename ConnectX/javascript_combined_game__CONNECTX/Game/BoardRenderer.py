from abc import ABCMeta, abstractmethod


class BoardRenderer(metaclass=ABCMeta):

    @abstractmethod
    def render(self, board):
        """
        The board rendering.
        :param board: The board.
        """