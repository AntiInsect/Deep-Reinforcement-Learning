from renju import RenjuBoard
import numpy as np

class Game(object):
    def __init__(self,player1,player2):
        self.player1 = player1
        self.player2 = player2
        self.board = RenjuBoard()

    def do_play(self):
        self.board.reset()
        states, mcts_probs = [], []
        while True:
            player = self.player2
            opponent = self.player1
            debug_stone = '◯'
            if self.board.get_current_player():
                player = self.player1
                opponent = self.player2
                debug_stone = '●'
            move, move_probs = player.get_action(self.board)

            # notice the opponent 
            opponent.notice(self.board,move) 

            # Resign 
            if move is None:
                end = True
                winner = (RenjuBoard.WHITE_WIN if self.board.get_current_player() else RenjuBoard.BLACK_WIN)
                # print ("player: ",debug_stone," resigns.")
            else:
                # store the data
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                # perform a move
                self.board.do_move_by_number(move)
                
                # open debug board which the game interface
                # print ("player: ", debug_stone)
                # self.board._debug_board()
                
                end, winner = self.board.game_end()



            if end:
                self.board._debug_board()
                total_moves = len(states)
                if winner == RenjuBoard.DRAW:
                    winner_map = [ 0 for _i in range(total_moves)]
                    print("draw")
                elif winner == RenjuBoard.WHITE_WIN:
                    winner_map = [ (_i%2) * 2 - 1 for _i in range(total_moves)]
                    print("WHITE_WIN")
                else:
                    winner_map = [ ((_i+1)%2)*2 - 1  for _i in range(total_moves)]
                    print("BLACK_WIN")
                return winner, zip(states, mcts_probs, winner_map)
