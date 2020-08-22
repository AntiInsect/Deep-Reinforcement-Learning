import Game.Board as BOARD
import Game.Game as Game
from console_select import select_player, select_network, set_AI_conf

from Player.Human import Human
from Player.AI_MCTS import AI_MCTS
from Player.AI_MCTS_Net import AI_MCTS_Net

from configure import Configure

from Game.BoardRenderer import BoardRenderer
import Game.Board as BOARD
from Game.Board import Board


class ConsoleRenderer(BoardRenderer):

    def render(self, board: Board):
        """
        Render the current board in console mode
        :param board:  The board.
        """
        # Build a 16 * 16 array.
        print_array = [["" for _ in range(BOARD.board_size + 1)] for _ in range(BOARD.board_size + 1)]

        index_string = [""] + [str(i) for i in range(BOARD.board_size)]

        # Row & Column Index.
        print_array[0] = index_string
        for row in range(BOARD.board_size):
            print_array[row + 1][0] = str(row)

        for i in range(BOARD.board_size):
            for j in range(BOARD.board_size):
                if board.board[i, j] == BOARD.o:
                    print_array[i + 1][j + 1] = "O"
                elif board.board[i, j] == BOARD.x:
                    print_array[i + 1][j + 1] = "X"
                else:
                    print_array[i + 1][j + 1] = "."

        # Print.
        for i in range(BOARD.board_size + 1):
            for j in range(BOARD.board_size + 1):
                print("{:^3}".format(print_array[i][j]), end="")
            print("")


def start():
    conf = Configure()
    conf.get_conf()

    def player_init(player_selected, name):
        if player_selected == 1:
            return Human(name=name)
        elif player_selected == 2:
            search_times, greedy_value = set_AI_conf(search_times=2000, greedy_value=5.0)
            return AI_MCTS(name=name,
                           search_times=search_times,
                           greedy_value=greedy_value,
                           is_output_analysis=conf.conf_dict["AI_is_output_analysis"])
        elif player_selected == 3:
            network = select_network()
            search_times, greedy_value = set_AI_conf(search_times=400, greedy_value=5.0)
            return AI_MCTS_Net(name=name,
                               policy_value_function=network.predict,
                               board_to_xlabel=network.board_to_xlabel,
                               is_training=False,
                               search_times=search_times,
                               greedy_value=greedy_value,
                               is_output_analysis=conf.conf_dict["AI_is_output_analysis"])

    player1_selected, name1 = select_player("Please input first player. Press <Ctrl-C> to end\n"
                                            "1: Human\n"
                                            "2: AI with pure Monte Carlo tree search\n"
                                            "3: AI with Monte Carlo tree search & neural network\n"
                                            ": ", allowed_input=[1, 2, 3])

    player1 = player_init(player1_selected, name1)

    player2_selected, name2 = select_player("Please input second player. Press <Ctrl-C> to end\n"
                                            "1: Human\n"
                                            "2: AI with pure Monte Carlo tree search\n"
                                            "3: AI with Monte Carlo tree search & neural network\n"
                                            ": ", allowed_input=[1, 2, 3])

    player2 = player_init(player2_selected, name2)

    console_renderer = ConsoleRenderer()

    print("############### Game Start ###############")
    winner = Game.start_until_game_over(player1, player2, board_renderer=console_renderer)
    if winner == BOARD.o:
        print("Congrats! \"O\" wins.")
    elif winner == BOARD.x:
        print("Congrats! \"X\" wins.")
    else:
        print("Draw!")
    print("############### Game Over ###############")


if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        exit(0)
