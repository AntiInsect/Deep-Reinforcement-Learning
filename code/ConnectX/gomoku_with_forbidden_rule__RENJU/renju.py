import numpy as np

class RenjuBoard(object):
    EMPTY_STONE = 0 
    BLACK_STONE = 1
    WHITE_STONE = 2
    WHITE_FIVE = 1
    BLACK_FIVE = 2
    BLACK_FORBIDDEN = 4

    WHITE_WIN = -1
    BLACK_WIN = 1
    DRAW = 0

    directions = {
        '|' : [[+1,0],[-1,0]],    # Bottom, Top
        '-' : [[0,+1],[0,-1]],    # Front, Back
        '\\' : [[+1,+1],[-1,-1]], # Right Bottom, Left Top
        '/' : [[+1,-1],[-1,+1]],  # Left Bottom, Right Top
    }

    @staticmethod
    def get_oppo(stone):
        return 3 - stone

    def __init__(self,init = ''):
        self.reset(init)

    def reset(self,init = ''):
        self.last_move = 0
        # self.availables = [i for i in range(225)]
        self.availables = [i for i in range(64)]

        # self.board = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
        self.board = [0,0,0,0,0,0,0,0,]
        self.current = [1,1]
        i = 0
    
        while i < len(init):
            self.do_move(init[i:i+2])
            i += 2

    def _debug_board(self):
        return_str = "\n "
        # for x in range(15):
        for x in range(8):
            return_str += "{:x}".format(x+1).center(4)
        return_str += "\r\n"
        # for i in range(1,16):
        for i in range(1,9):
            return_str += "{:x}".format(i)
            # for j in range(1,16):
            for j in range(1,9):
                if self._([i,j]) == RenjuBoard.BLACK_STONE:
                    return_str += '●'.center(3)
                elif self._([i,j]) == RenjuBoard.WHITE_STONE:
                    return_str += '◯'.center(3)
                else:
                    return_str += '-'.center(4)
            return_str += '\r\n\r\n'
        print (return_str)

    @staticmethod
    def pos2coordinate(position):
        return [
            # int(position[0],16),
            # int(position[1],16),
            int(position[0],9),
            int(position[1],9),
        ]

    @staticmethod
    def coordinate2pos(coordinate):
        return "{:x}{:x}".format(coordinate[0],coordinate[1])

    @staticmethod
    def pos2number(position):
        # return (int(position[0], 16) - 1) * 15 + int(position[1],16) - 1
        return (int(position[0], 9) - 1) * 8 + int(position[1],9) - 1

    @staticmethod
    def number2pos(num):
        # return "{:x}{:x}".format((num // 15) + 1,(num % 15) + 1)
        return "{:x}{:x}".format((num // 8) + 1,(num % 8) + 1)

    @staticmethod
    def num2coordinate(num):
        return [
            # (num // 15) + 1 ,
            # (num % 15) + 1 ,
            (num // 8) + 1 ,
            (num % 8) + 1 ,
        ]

    @staticmethod
    def coordinate2number(coordinate):
        # return (coordinate[0] - 1) * 15 + coordinate[1] - 1
        return (coordinate[0] - 1) * 8 + coordinate[1] - 1
 
    def do_move(self,pos):
        num = RenjuBoard.pos2number(pos)
        self.do_move_by_number(num)

    def do_move_by_number(self,number):
        coor = RenjuBoard.num2coordinate(number)
        color = RenjuBoard.WHITE_STONE
        if self.get_current_player():
            color = RenjuBoard.BLACK_STONE
        self.setStone(color,coor)
        self.last_move = number
        self.availables.remove(number)


    # def _move_to(self, to=[8,8]):
        # if to[0] >= 1 and to[0] <= 15 and to[1] >= 1 and to[1] <= 15:
    def _move_to(self, to=[4,4]):
        if to[0] >= 1 and to[0] <= 8 and to[1] >= 1 and to[1] <= 8:
            self.current = to
        return self._()

    def setStone(self,stone = 0,coordinate = []):
        if len(coordinate) == 0:
            coordinate = self.current
        
        row = self.board[coordinate[0] -1]
        row &= ~(1 << (coordinate[1] -1))
        # row &= ~(1 << (coordinate[1] -1 + 16))
        row &= ~(1 << (coordinate[1] -1 + 9))
        if stone == self.WHITE_STONE:
            # row |= (1 << (coordinate[1] -1 + 16))
            row |= (1 << (coordinate[1] -1 + 9))
        elif stone == self.BLACK_STONE:
            row |= (1 << (coordinate[1] -1))
        self.board[coordinate[0] -1] = row

    def moveDirection(self,direction):
        next = [
            self.current[0] + direction[0],
            self.current[1] + direction[1],
        ]
        # if next[0] < 1 or next[0] > 15 or next[1] < 1 or next[1] > 15:
        #     return False
        if next[0] < 1 or next[0] > 8 or next[1] < 1 or next[1] > 8:
            return False
        self.current = next
        return self._()

    def _(self,coordinate = []):
        if len(coordinate) == 0:
            coordinate = self.current
        if self.board[coordinate[0] - 1] & (1 << coordinate[1] - 1) :
            return self.BLACK_STONE
        # if self.board[coordinate[0] - 1] & (1 << coordinate[1] - 1 + 16):
        #     return self.WHITE_STONE
        if self.board[coordinate[0] - 1] & (1 << coordinate[1] - 1 + 9):
            return self.WHITE_STONE
        return self.EMPTY_STONE

    def count_stone(self,coordinate,shape):
        color = self._(coordinate)
        if color == RenjuBoard.BLACK_STONE or color == RenjuBoard.WHITE_STONE:
            count = 1
            for direction in RenjuBoard.directions[shape]:
                self._move_to(coordinate)
                while color == self.moveDirection(direction):
                    count = count + 1
            return count
        return 0

    def isFive(self,coordinate,color,shape = '',rule = 'renju'):
        if self._(coordinate) != RenjuBoard.EMPTY_STONE:
            return False
        self.setStone(color,coordinate)
        result = False
        if shape:
            count = self.count_stone(coordinate,shape)
            result = self.count_as_five(count,color,rule)
        else:
            for s in RenjuBoard.directions.keys():
                count = self.count_stone(coordinate,s)
                result = self.count_as_five(count,color,rule)
                if result:
                    break
        self.setStone(RenjuBoard.EMPTY_STONE,coordinate)
        return result

    def isFour(self,coordinate, color, shape = ''):
        if shape == '':
            for s in RenjuBoard.directions.keys():
                result,defense_point = self.isFour(coordinate, color, s)
                if result :
                    return result,defense_point
            return result,defense_point
        defense_point = None
        if self._(coordinate) != RenjuBoard.EMPTY_STONE:
            return False,defense_point
        result = 0
        self.setStone(color,coordinate)
        count_stone = 1
        for direction in RenjuBoard.directions[shape]:
            self._move_to(coordinate)
            while color == self.moveDirection(direction):
                count_stone = count_stone + 1
            current_Stone_copy = self.current.copy()

            if self.isFive(self.current,color,shape):
                result = result + 1
                defense_point = current_Stone_copy.copy()
        if count_stone == 4 and result == 2:
            result = 1
        self.setStone(RenjuBoard.EMPTY_STONE,coordinate)
        return result,defense_point

    def isOpenFour(self,coordinate,shape = '|'):
        if self._(coordinate) != RenjuBoard.EMPTY_STONE:
            return False
        count_active = 0
        self.setStone(RenjuBoard.BLACK_STONE,coordinate)
        count_black = 1
        for direction in RenjuBoard.directions[shape]:
            self._move_to(coordinate)
            while RenjuBoard.BLACK_STONE == self.moveDirection(direction):
                count_black = count_black + 1
            if self.isFive(self.current,RenjuBoard.BLACK_STONE,shape):
                count_active = count_active + 1
            else:
                break
        self.setStone(RenjuBoard.EMPTY_STONE, coordinate)
        if count_black == 4 and count_active == 2:
            if self.isForbidden(coordinate):
                return False
            return True
        return False

    def isOpenThree(self, coordinate, shape='|'):
        result = False
        self.setStone(RenjuBoard.BLACK_STONE, coordinate)
        for direction in RenjuBoard.directions[shape]:
            self._move_to(coordinate)
            while RenjuBoard.BLACK_STONE == self.moveDirection(direction):
                None

            if self._() == RenjuBoard.EMPTY_STONE:
                if self.isOpenFour(self.current,shape):
                    result = True
                    break
            else:
                break
        self.setStone(RenjuBoard.EMPTY_STONE,coordinate)
        return result

    def isDoubleThree(self, coordinate):
        count = 0
        for s in RenjuBoard.directions.keys():
            if self.isOpenThree(coordinate, s):
                count = count + 1
                if count >= 2:
                    return True
        return False

    def isDoubleFour(self,coordinate):
        count = 0
        for s in RenjuBoard.directions.keys():
            count_four, defense = self.isFour(
                coordinate, RenjuBoard.BLACK_STONE, s)
            count += count_four
            if count >= 2:
                return True
        return False

    def isOverline(self,coordinate):
        self.setStone(RenjuBoard.BLACK_STONE,coordinate)
        result = False
        for s in RenjuBoard.directions.keys():
            if self.count_stone(coordinate,s) > 5:
                result = True
                break
        self.setStone(RenjuBoard.EMPTY_STONE,coordinate)
        return result

    def count_as_five(self,number,color,rule = 'renju'):
        if color == RenjuBoard.WHITE_STONE and rule == 'renju':
            return number >= 5
        return number == 5

    def isForbidden(self,coordinate):
        if self._(coordinate) != RenjuBoard.EMPTY_STONE:
            return False
        if self.isFive(coordinate,RenjuBoard.BLACK_STONE):
            return False
        return (self.isOverline(coordinate) or self.isDoubleFour(coordinate) or self.isDoubleThree(coordinate))

    def Find_win(self):
        attacker = (RenjuBoard.BLACK_STONE if self.get_current_player() else RenjuBoard.WHITE_STONE)
        defender = RenjuBoard.get_oppo(attacker)
        oppo_win = None
        oppo_win_count = 0
        # for i in range(1,16):
        #         for j in range(1,16):
        for i in range(1,9):
                for j in range(1,9):
                    if self.isFive([i,j],attacker):
                        return RenjuBoard.coordinate2number([i,j]),None,0
                    if self.isFive([i,j],defender):
                        oppo_win = RenjuBoard.coordinate2number([i,j])
                        oppo_win_count += 1
        return None,oppo_win,oppo_win_count

    
    def VCF(self):
        vcf_path = []
        return_str = ''
        expands = []
        win = False
        win_by_forbidden = False

        # attacker: try to get an open four
        # defender: try to stop him
        attacker = (RenjuBoard.BLACK_STONE if self.get_current_player() else RenjuBoard.WHITE_STONE)
        defender = RenjuBoard.get_oppo(attacker)
        
        def expand_vcf(board): 
            collect = []
            oppo_win = [] 
            # for i in range(1,16):
            #     for j in range(1,16):
            for i in range(1,9):
                for j in range(1,9):
                    if board.isFive([i,j],attacker):
                        return [i,j],[]
                    elif board.isFive([i,j],defender):
                        oppo_win.append([i,j])
                    count_four,defense = board.isFour([i,j],attacker)
                    if count_four > 0 and (attacker == RenjuBoard.WHITE_STONE or not board.isForbidden([i,j])):
                        collect.append([ [i,j] , defense])
            if len(oppo_win) > 1:
                return False,[] 
            if len(oppo_win) == 1:
                for _c in collect:
                    if oppo_win[0] == _c[0]:
                        return False,[ _c.copy() ]
                return False,[]
            return False,collect

        while True:
            win , availables = expand_vcf(self)
            if win :
                break
            if len(vcf_path) == 0 and len(availables) == 0:
                return None , ''
            expands.append(availables)

            while True:
                if len(expands) == 0:
                    break
                not_expanded = expands.pop()
                if(len(not_expanded) > 0):
                    break
                elif len(vcf_path) > 0:
                    to_remove = vcf_path.pop()
                    self.setStone(RenjuBoard.EMPTY_STONE,to_remove[0])
                    self.setStone(RenjuBoard.EMPTY_STONE,to_remove[1])

            if len(expands) == 0 and len(not_expanded) == 0:
                break
            next_try = not_expanded.pop()
            vcf_path.append(next_try)
            expands.append(not_expanded)
            if defender == RenjuBoard.BLACK_STONE and self.isForbidden(next_try[1]):
                win = next_try[1]
                win_by_forbidden = True
                break
            self.setStone(attacker,next_try[0])
            self.setStone(defender,next_try[1])
        if win:
            for move_pair in vcf_path:
                return_str += RenjuBoard.coordinate2pos(move_pair[0])
                return_str += RenjuBoard.coordinate2pos(move_pair[1])
                self.setStone(RenjuBoard.EMPTY_STONE,move_pair[0])
                self.setStone(RenjuBoard.EMPTY_STONE,move_pair[1])
            if not win_by_forbidden:
                return_str += RenjuBoard.coordinate2pos(win)
        win_move = None
        if return_str:
            win_move = RenjuBoard.pos2number(return_str[0:2])
        return win_move , return_str

    def GetResult(self,player):
        is_end, winner = self.game_end()
        if is_end:
            if winner == RenjuBoard.DRAW:
                return RenjuBoard.DRAW
            if (player == 1 and winner == RenjuBoard.BLACK_WIN) or (player == 0 and winner == RenjuBoard.WHITE_WIN):
                return 1
            else:
                return 0
        return 0
        
    def game_end(self):
        coordinate = RenjuBoard.num2coordinate(self.last_move)
        color = RenjuBoard.BLACK_STONE
        if self.get_current_player():
            color = RenjuBoard.WHITE_STONE
        self.setStone(RenjuBoard.EMPTY_STONE,coordinate)
        is_end, winner = self.checkWin(coordinate,color)
        self.setStone(color,coordinate)
        return is_end, winner

    def checkWin(self,coordinate,color):
        #coordinate = self.pos2coordinate(position)
        if color == RenjuBoard.WHITE_STONE:
            if self.isFive(coordinate,color):
                return True,RenjuBoard.WHITE_WIN
        else:
            if self.isFive(coordinate,color):
                return True,RenjuBoard.BLACK_WIN
            if self.isForbidden(coordinate):
                return True,RenjuBoard.WHITE_WIN
        
        counting = 0
        for row in self.board:
            # row_white = row >> 16
            row_white = row >> 9
            row_black = row & 131071
            row_stones = row_black | row_white
            row_empty = 131071 - row_stones
            while row_empty > 0:
                if row_empty % 2:
                    counting += 1
                    if counting > 1:
                        return False , -1
                row_empty = row_empty >> 1
            
        return True, RenjuBoard.DRAW

    def gomokuCheckWin(self,coordinate,color):
        #coordinate = self.pos2coordinate(position)
        if self.isFive(coordinate,color,'','gomoku'):
            if color == RenjuBoard.BLACK_STONE:
                return RenjuBoard.BLACK_WIN
            else:
                return RenjuBoard.BLACK_WIN
        return False

    def get_current_player(self):
        return len(self.availables) % 2

    def dump_black(self):
        dump_map = []
        # for i in range(1,16):
        #     row = []
        #     for j in range(1,16):
        for i in range(1,9):
            row = []
            for j in range(1,9):
                if self._([i,j]) == RenjuBoard.BLACK_STONE:
                    row.append(1)
                else:
                    row.append(0)
            dump_map.append(row)
        return dump_map

    def dump_white(self):
        dump_map = []
        # for i in range(1,16):
        #     row = []
        #     for j in range(1,16):
        for i in range(1,9):
            row = []
            for j in range(1,9):
                if self._([i,j]) == RenjuBoard.WHITE_STONE:
                    row.append(1)
                else:
                    row.append(0)
            dump_map.append(row)
        return dump_map

    def dump_empty(self):
        dump_map = []
        # for i in range(1,16):
        #     row = []
        #     for j in range(1,16):
        for i in range(1,9):
            row = []
            for j in range(1,9):
                if self._([i,j]) == RenjuBoard.EMPTY_STONE:
                    row.append(1)
                else:
                    row.append(0)
            dump_map.append(row)
        return dump_map

    def dump_forbiddens(self):
        dump_map = []
        # for i in range(1,16):
        #     row = []
        #     for j in range(1,16):
        for i in range(1,9):
            row = []
            for j in range(1,9):
                if self.isForbidden([i,j]):
                    row.append(1)
                else:
                    row.append(0)
            dump_map.append(row)
        return dump_map

    def dump_win_points(self):
        dump_map = []
        # for i in range(1,16):
        #     row = []
        #     for j in range(1,16):
        for i in range(1,9):
            row = []
            for j in range(1,9):
                if self.isFive([i,j],RenjuBoard.BLACK_STONE) or self.isFive([i,j],RenjuBoard.WHITE_STONE):
                    row.append(1)
                else:
                    row.append(0)
            dump_map.append(row)
        return dump_map

    
    def current_state(self):
        # square_state = np.zeros((3, 15, 15))
        # for i in range(15):
        #     for j in range(15):
        #         ijstone = self._([i+1,j+1])
        square_state = np.zeros((3, 8, 8))
        for i in range(8):
            for j in range(8):
                ijstone = self._([i+1,j+1])
                if ijstone == RenjuBoard.BLACK_STONE:
                    square_state[0][i][j] = 1
                elif ijstone == RenjuBoard.WHITE_STONE:
                    square_state[1][i][j] = 1
        square_state[2][:, :] = self.get_current_player()
        return square_state

if __name__ == "__main__":
    # testboard = RenjuBoard('8889878698789a76979979a696a7aaa4a89577847346')
    # testboard = RenjuBoard('8889878698789a76979979a696a7aaa4a895')
    testboard = RenjuBoard('44546463656655')
    testboard._debug_board()
    print(testboard.VCF())