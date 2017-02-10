from __future__ import print_function, division

__author__ = 'amrit'

from random import randint, random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
from config import LEARNERS
from test import split_two, cut_position, divide_train_test

sys.dont_write_bytecode = True

def execute(l, samples=np.array([]), labels=[]):
    return balance(samples, labels,m=int(l[0]), r=int(l[1]), neighbors=int(l[2]))

def smote(data, num, k=5,r=1):
    corpus = []
    if len(data)<k:
        k=len(data)-1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k-1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))
    return corpus

def balance(data_train, train_label, m=0, r=0, neighbors=0):
    pos_train = []
    neg_train = []
    for j, i in enumerate(train_label):
        if i == 1:
            pos_train.append(data_train[j])
        else:
            neg_train.append(data_train[j])
    pos_train = np.array(pos_train)
    neg_train = np.array(neg_train)

    if len(pos_train) < len(neg_train):
        pos_train = smote(pos_train, m, k=neighbors,r=r)
        if len(neg_train) < m:
            m = len(neg_train)
        neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
    #print(pos_train,neg_train)
    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)
    return data_train1, label_train

def main(*x, **r):

    l = np.asarray(x)

    split = split_two(np.array(r['data_samples']), np.array(r['target']))
    pos = np.array(split['pos'])
    neg = np.array(split['neg'])

    ## 20% train and grow
    cut_pos, cut_neg = cut_position(pos, neg, percentage=80)
    data_train, train_label, data_test, test_label = divide_train_test(pos, neg, cut_pos, cut_neg)
    data_train,train_label=execute(l, samples=data_train, labels=train_label)
    learner=LEARNERS[r['learner']]()
    return learner.run(data_train, train_label, data_test, test_label),0


