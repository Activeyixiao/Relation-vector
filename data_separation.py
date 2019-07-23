import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from collections import ChainMap
import random
import networkx as nx
from argparse import ArgumentParser
import os
import sys

def graph_generation(graph, wp_ls, node):
    G = graph
    wp_train = set()
    finished_type = set()    
    for a in list(G.neighbors(node)):
        wp_train.add(str(node+'__'+a))
        finished_type.add(node)
        if a not in finished_type:
            finished_type.add(a)
            for b in list(G.neighbors(a)):
                wp_train.add(str(a+'__'+b))
                if b not in finished_type:
                    finished_type.add(b)
                    for c in list(G.neighbors(b)):
                        wp_train.add(str(b+'__'+c))
    return wp_train

def data_separatation(train, wp_ls):
    wp_test = list(set(wp_ls) - train)
    w_mentioned_train = set()
    for wp in train:
        w1,w2 = wp.split('__')
        w_mentioned_train.add(w1)
        w_mentioned_train.add(w2)       
    wp_test_refine = []
    for i in wp_test:
        if i.split('__')[0] not in w_mentioned_train and i.split('__')[1] not in w_mentioned_train:
            wp_test_refine.append(i)
    remain = list(set(wp_test) - set(wp_test_refine))
        
    w_mentioned_test = set()
    for wp in wp_test_refine:
        w1,w2 = wp.split('__')
        w_mentioned_test.add(w1)
        w_mentioned_test.add(w2)   
    back_train_list = []
    for i in remain:
        if i.split('__')[0] not in w_mentioned_test and i.split('__')[1] not in w_mentioned_test:
            back_train_list.append(i)
    wp_train_refine = list(train) + back_train_list
    remain_ls = list(set(remain) - set(back_train_list))
    return wp_train_refine,wp_test_refine, remain_ls

def utilization_rate(train, test, remain):
    lost_wp = len(remain)
    rate = len(train)/len(test) 
    return lost_wp, rate

def find_best_nodes(graph, wordpair_ls):
    wp_ls = wordpair_ls
    G = graph
    degree_ls = (sorted(G.degree, key=lambda x: x[1], reverse=True))
    high_degree_nodes = [i[0] for i in degree_ls if 60>i[1]>50]
    panel = []
    for node in high_degree_nodes:
        WP_train = graph_generation(G,wp_ls,node)
        train, test, remain = data_separatation(WP_train, wp_ls)
        lost_score, rate_score  = utilization_rate(train, test, remain)
        if 1.5 < rate_score < 3.0:
            panel.append((lost_score,rate_score,node))
        print('finished the relation type'+' '+node+' '+'which lost '+str(lost_score)+' wp and has'+str(rate_score)+'train test rate'  )
    rank = sorted(panel, key=lambda x:x[0])
    return rank

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-b','--experiment-data', help='Build folder', required=True)
    parser.add_argument('-r','--relation-vectors', help='The relation vectors from corpus', required=True)

    args = parser.parse_args()
    wordpair_ls = []
    with open(args.relation_vectors, "r") as input_f:
        for line in input_f:
            wp = line.strip().split(' ')[0]
            wordpair_ls.append(wp)
    wordpair_ls.remove(wordpair_ls[0])
    print('get all the words pairs in the file')

    DG = nx.DiGraph()
    for wp in wordpair_ls:
        w1 = wp.split('__')[0]
        w2 = wp.split('__')[1]
        DG.add_edge(w1,w2)
    print('get directed graph')


    # ranked_nodes = find_best_nodes(DG, wordpair_ls)
    # best_node = ranked_nodes[0][2]
    # print('the best node for graph separation is '+ best_node)
    part_of_train = graph_generation(DG, wordpair_ls, 'rake')
    train_data,test_data,remain_data = data_separatation(part_of_train, wordpair_ls)

    path=args.experiment_data
    if not os.path.exists(path):
        os.makedirs(path)

    output_path_train = os.path.join(path,'train_data '+str(len(train_data))+'.vec')
    output_path_test = os.path.join(path,'test_data '+ str(len(train_data))+'.vec')
    output_path_remain = os.path.join(path,'remain_data '+str(len(remain_data))+'.vec')

    with open(output_path_train,'w') as train_output:
        train_output.write(str(len(train_data))+' '+str(1838)+'\n')
        with open(output_path_test,'w') as test_output:
            test_output.write(str(len(test_data))+' '+str(1838)+'\n')
            with open(output_path_remain,'w') as remain_output:
                remain_output.write(str(len(remain_data))+' '+str(1838)+'\n')
                with open(args.relation_vectors,'r') as input_f:
                    for line in input_f:
                        wp = line.strip().split(' ')[0]
                        if wp in train_data:
                            train_output.write(line)
                        elif wp in test_data:
                            test_output.write(line)
                        else:
                            remain_output.write(line)