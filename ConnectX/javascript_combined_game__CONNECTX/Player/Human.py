from Game.Board import Board
from Player.Player import Player


class Human(Player):

    def __init__(self, name="Human"):
        self.name = name

    def __str__(self):
        return "-----  Human -----"

    def take_action(self, board: Board, is_output_action=True, running_output_function=None, is_stop=None):
        """
        It's turn to you.
        :param board: Current board.
        :param is_output_action:  Whether to output action information.
        :param running_output_function: running output function.
        :param is_stop:  Ask whether to stop.
        :return: <tuple (i, j)>  Coordinate of the action.
        """
        print(" It's turn to {0}, human player.".format(self.name))
        while True:
            # Input.
            input_str = input(
                "Please input the coordinates {0} wants to move, "
                "the format is \"[Row],[Column]\":\n".format(self.name))

            # Validate.
            try:
                if input_str.isdigit():
                    print("Please enter full coordinates.\n")
                    continue
                action = [int(index) for index in input_str.split(",")]
            except:
                print("The input format is incorrect. Please try again.\n")
                continue

            # 执行。 Execute.
            if not board.step(action):
                print("Cannot move here. Please try again.\n")
                continue

            print("Human player {0} moves ({1}, {2})\n".format(self.name, action[0], action[1]))
            break

        return action
