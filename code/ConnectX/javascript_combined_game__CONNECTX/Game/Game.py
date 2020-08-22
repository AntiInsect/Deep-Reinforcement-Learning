from Player.Player import Player
from Game.Board import Board
import Game.Board as BOARD
from Game.BoardRenderer import BoardRenderer


def start_until_game_over(player1: Player, player2: Player, board_renderer: BoardRenderer = None):
    """
    Player player1 and player2 play on the board until the game is over, and output the winner.
    :param player1: Player 1.
    :param player2: Player 2.
    :param board_renderer: The board renderer.
    :return: <int> board The winner returned by board.
    """
    board = Board()
    while True:
        # Render.
        if board_renderer is not None:
            board.render(board_renderer)

        # Take action.
        if board.current_player == BOARD.o:
            player1.take_action(board, is_output_action=board_renderer is not None)
        else:
            player2.take_action(board, is_output_action=board_renderer is not None)

        # Game over?
        is_over, winner = board.result()
        if is_over:
            if board_renderer is not None:
                board.render(board_renderer)
            return winner
