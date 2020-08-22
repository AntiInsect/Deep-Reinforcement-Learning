import numpy as np
from mcts_alphaZero import MCTS
from renju import RenjuBoard


class Human(object):
    def __init__(self):
        None
    
    def get_action(self, board):
        location = input("Your move: (11 to 88)")
        # You can resign
        if location == 'RESIGN':
            return None, None
        move_number = RenjuBoard.pos2number(location.strip())
        if move_number not in board.availables:
            print("invalid move")
            location = self.get_action(board)
        # prob = np.zeros(15*15)
        prob = np.zeros(8*8)
        prob[move_number] = 1.0
        return move_number,prob

    def notice(self,board,move):
        pass

    def __str__(self):
        return "Human"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=1000, is_selfplay=0, debug=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, debug = debug)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.reset()

    def notice(self,board,move):
        # print ("been told move ", move)
        self.mcts.update_with_move(board, move)

    def get_action(self, board, temp=1e-3):
        #sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        if self._is_selfplay:
            temp = 1.5
        # move_probs = np.zeros(15*15)
        move_probs = np.zeros(8*8)
        acts, probs = self.mcts.get_move_probs(board, temp)

        # The AI can also resign if it does not give the action and probs
        if acts is None:
            return None, None

        move_probs[list(acts)] = probs
        best_chance = np.max(move_probs)
        best_move = np.where(move_probs == best_chance)[0][0]
        if self._is_selfplay:
            move = np.random.choice(
                acts,
                # p = probs
                p = 0.9*probs + 0.1*np.random.dirichlet(0.3*np.ones(len(probs)))
            )
            # open debug
            # print("choose ", RenjuBoard.number2pos(move) ,"by prob ", move_probs[move])
            # print("best move is ", RenjuBoard.number2pos(best_move), "by prob ", best_chance)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            # move = np.random.choice(acts, p=probs)
            move = best_move
            # reset the root node
            # self.mcts.update_with_move(-1)
        self.mcts.update_with_move(board, move)
        return move, move_probs


class PolicyPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function):
        self.policy_fn = policy_value_function

    def get_action(self, board):
        # move_probs = np.zeros(15 * 15)
        move_probs = np.zeros(8 * 8)
        action_probs, leaf_value = self.policy_fn(board)
        for action, prob in action_probs:
            move_probs[action] = prob
        best_chance = np.max(move_probs)
        best_move = np.where(move_probs == best_chance)[0][0]
        return best_move, move_probs

    def notice(self,board,move):
        pass

    def __str__(self):
        return "With Given Policy"
