#!/usr/bin/env python
# coding: utf-8

from gobang import evaluation,searcher
from learn import Field,QLearning
import pickle
import random

b=Field()  
QL = QLearning(b)
output1 = open('output1.txt', 'rb')
history = pickle.load(output1)  
QL.add_record(history)
record2 = []
record1 = []
e = evaluation()
s = searcher()
q = True

while q:
    row_2 = input('Enter row (player 2) ')
    col_2 = input('Enter column(player 2) ')
    record2.append(((int(row_2)),(int(col_2))))
    b.add_record2(record2)
    b.set_field_data()
    s.board=b.field_data.board()
    best_actions = []
    value = e.evaluate(b.field_data,2)
    print(value)
    if value<-9900:
        score,row,col=s.search(1,1)
        choice = (row,col)
        print("About to Win")
    if value>9900:
        score,row,col=s.search(1,1)
        choice = (row,col)
        print("About to Lose")           
    else:
        
        for idx in [(_x,_y) for _x,_y in ([(7,7)]+record1+record2)]:
            for a in b.get_actions(idx):
                max_q_value = -10000000000000000000
                try:
                    q_value = QL.Qvalue[(int(row_2),int(col_2))][a]
                    if (q_value>max_q_value) and not(a in record2) and not(a in record1):
                        best_actions = [a,]
                        max_q_value = q_value
                    elif (q_value == max_q_value) and not(a in record2) and not(a in record1):
                        best_actions.append(a)
                        best_actions = list(set(best_actions))
                    choice = random.choice(best_actions)
                except KeyError:
                    choice = random.choice(b.get_actions((int(row_2),int(col_2))))
    record1.append(choice)
    print([choice[0]+1,choice[1]+1])
    b.add_record1(record1)
    b.set_field_data()
    b.display()
    