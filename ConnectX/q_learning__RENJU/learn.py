#!/usr/bin/env python
# coding: utf-8

import copy
import random
from gobang import chessboard, evaluation, searcher
import pickle


ALPHA = 0.5
GAMMA  = 0.8 
E_GREEDY_RATIO = 0.01
LEARNING_COUNT = 1000000
iii = 0
search_id = 0
record1 = []
record2 = []
not_finished = True

class Field(object):

    def __init__(self):
        self.record1=[(7,7)]
        self.record2=[]
        self.set_field_data()
        self.iii = iii   

    def add_record1(self,x):
        self.record1 += x

    def add_record2(self,x):
        self.record2 += x
        
    def set_field_data(self):
        b = chessboard()
        self.field_data = b
        for idx,val in enumerate(self.record1):
            self.field_data[val[0]][val[1]] = 1
        for idx,val in enumerate(self.record2):
            self.field_data[val[0]][val[1]] = 2
            
    def display(self, point=None):
        field_data = copy.deepcopy(self.field_data.board())
        if not point is None:
                x, y = point
                field_data[x][y] = "@"
        else:
            point = ""

        print("----- Dump Field: %s -----" % str(point))
        for line in field_data:
                print("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, point):
        x,y = point
        around_map = [(x-1,y-1), (x-1 , y),(x-1,y+1), (x, y-1),(x, y+1),(x+1,y-1),(x+1, y),(x+1,y+1)]
        if (x!=0) and (x!=14) and (y!=0) and (y!=14):
            return [(_x, _y) for _x, _y in around_map if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
        if (x==0)and(y!=14)and(y!=0):
            return [(_x, _y) for _x, _y in around_map[3:] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
        if (x==14)and(y!=14)and(y!=0): 
            return [(_x, _y) for _x, _y in around_map[:5] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))] 
        if (y==0)and(x!=0)and(x!=14):
            return [(_x, _y) for _x, _y in (around_map[1:3]+around_map[4:5]+around_map[6:7]) if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
        if (y==14)and(x!=0)and(x!=14):
            return [(_x, _y) for _x, _y in (around_map[:2]+around_map[3:4]+around_map[5:7]) if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
        else:
            if (x==14)and(y==14):
                return [(_x, _y) for _x, _y in [(x-1,y),(x,y-1),(x-1,y-1)] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
            if (x==0)and(y==14):
                return [(_x, _y) for _x, _y in [(x,y-1),(x+1,y-1),(x+1,y-1)] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
            if (x==14)and(y==0):
                return [(_x, _y) for _x, _y in [(x-1,y),(x,y+1),(x-1,y+1)] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
            if (x==0)and(y==0):
                return [(_x, _y) for _x, _y in [(x+1,y),(x+1,y+1),(x,y+1)] if ((self.field_data[_x][_y] != 1) and((self.field_data[_x][_y] != 2)))]
                
    def get_val(self, point,i):
        if i == 2: j = 1
        if i == 1: j = 2
        
        x, y = point
        e = evaluation()
        s = searcher()
        field_data = copy.deepcopy(self.field_data.board())
        s.board = field_data
        field_data[x][y] = j
        score,row,col = s.search(i,2)
        field_data[row][col]=i
        v = e.evaluate(field_data,j)-e.evaluate(field_data,i)
        self.iii += 1
        if (self.field_data.check()==0):
            return v, False,(row,col)
        else:
            print("one round finished winner " + str(self.field_data.check()))
            self.iii = 0
            return v, True, (row, col)
        
    def get_start_point(self,first):
        if first:
            point = [(6,6),(6,7),(6,8),(7,6),(7,8),(8,6),(8,7),(8,8)][random.choice([0,1,2,3,4,5,6,7])]
            self.add_record2([point])
            self.set_field_data()
            return point
        else:
            s = searcher()
            field_data = copy.deepcopy(self.field_data.board())
            s.board = field_data
            score,row,col = s.search(2,2)
            self.add_record2([(row,col)])
            self.set_field_data()
            return (row,col)
            
            
class QLearning(object):
    """ class for Q Learning """

    def __init__(self, map_obj):
        self.Qvalue = {}
        self.Field = map_obj

    def add_record(self,x):
        self.Qvalue = x

    def learn(self, greedy_flg=False):
        print("----- Episode -----")
        state = self.Field.get_start_point(True)
        
        for _ in range(100):
            if greedy_flg:
                action = self.choose_action_greedy(state)
                self.Field.display(action)
                print("\tstate: %s -> action:%s\n" % (state, action))

            if E_GREEDY_RATIO < random.random():
                aiu = []
                for idx in [(_x,_y) for _x,_y in (self.Field.record1+self.Field.record2)]:
                    for a in b.get_actions(idx):
                        aiu.append(a)
                choice = random.choice(aiu)
            else:
                for idx in [(_x,_y) for _x,_y in (self.Field.record2+self.Field.record1)]:
                    for a in b.get_actions(idx):
                        max_q_value = -10000000000000000000
                        q_value = self.get_Qvalue(state, a)
                        if (q_value>max_q_value):
                            best_actions = [a,]
                            max_q_value = q_value
                        elif (q_value == max_q_value):
                            best_actions.append(a)
                            best_actions = list(set(best_actions))
                choice = random.choice(best_actions)

            action = choice
            self.update_Qvalue(state,action)
            self.Field.add_record1([action])
            self.Field.set_field_data()
            state = self.Field.get_start_point(False)

            if self.Field.get_val(action,2)[1]:
                break

    def update_Qvalue(self, state, action):
        Q_s_a = self.get_Qvalue(state, action)
        if len(self.Field.get_actions(action)) != 0:
            mQ_s_a = max([self.get_Qvalue(action, n_action) for n_action in self.Field.get_actions(action)])
            r_s_a, finish_flg, coord = self.Field.get_val(action, 2)
            q_value = Q_s_a + ALPHA * ( r_s_a +  GAMMA * mQ_s_a - Q_s_a)
            self.set_Qvalue(state, action, q_value)
            return finish_flg
        else:
            return True

    def get_Qvalue(self, state, action):
        try:
            return self.Qvalue[state][action]
        except KeyError:
            return 0.0

    def set_Qvalue(self, state, action, q_value):
        print([state,action])
        self.Qvalue.setdefault(state,{})
        self.Qvalue[state][action] = q_value

    def choose_action(self, state):
        if E_GREEDY_RATIO < random.random():
            return random.choice(self.Field.get_actions(state))
        else:
            return self.choose_action_greedy(state)

    def choose_action_greedy(self, state):
        best_actions = []
        max_q_value = -10000000000000000000
        for a in self.Field.get_actions(state):
                q_value = self.get_Qvalue(state, a)
                if q_value > max_q_value:
                        best_actions = [a,]
                        max_q_value = q_value
                elif q_value == max_q_value:
                        best_actions.append(a)
                best_actions = list(set(best_actions))
        choice = random.choice(best_actions)
        return choice

    def dump_Qvalue(self):
        print("##### Dump Qvalue #####")
        for i, s in enumerate(self.Qvalue.keys()):
            for a in self.Qvalue[s].keys():
                    print("\t\tQ(s, a): Q(%s, %s): %s" % (str(s), str(a), str(self.Qvalue[s][a])))
            if i != len(self.Qvalue.keys())-1:
                print('\t----- next state -----')



if __name__ == "__main__":
    a = {}
    for i in range(LEARNING_COUNT):
            print("----- Episode "+str(i)+"-----")
            b=Field()
            
            QL = QLearning(b)
            QL.add_record(a)
            QL.learn()
            a = QL.Qvalue
            E_GREEDY_RATIO = E_GREEDY_RATIO - (0.1/LEARNING_COUNT)
    pickle.dump( a, open( "output1.txt", "wb" ) )
