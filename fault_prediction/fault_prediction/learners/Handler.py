from __future__ import print_function

from random import shuffle, randint, random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import sys

sys.dont_write_bytecode = True


class Handler(object):
    """
        1) This class is the entry point to any learners.
        2)
    """
    def __init__(self, data, result):
        super(Handler, self).__init__()
        self.result = result
        self.data=data

    def execute(self,l, samples=[], labels=[]):
        return self.balance(samples, labels, m=int(l[0]), r=int(l[1]), neighbors=int(l[2]))

    def smote(self,data, num, k=5, r=1):
        corpus = []
        if len(data) < k:
            k = len(data) - 1
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

    def balance(self,data_train, train_label, m=0, r=0, neighbors=0):
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
            pos_train = self.smote(pos_train, m, k=neighbors, r=r)
            if len(neg_train) < m:
                m = len(neg_train)
            neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
        data_train1 = np.vstack((pos_train, neg_train))
        label_train = [1] * len(pos_train) + [0] * len(neg_train)
        return data_train1, label_train

    def run(self):
        pass
