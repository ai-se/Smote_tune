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
    def __init__(self, data, result, folds=5, splits=5):
        super(Handler, self).__init__()
        self.folds = folds
        self.splits = splits
        self.data = data
        self.result = result

    def execute(self,smote_val):
        """
         :param folds: number of folds in cross-validations. Default is 2.
         :param splits: number of splits in cross-validations. Default is 2.
        """

        for _ in range(self.folds):
            shuffle(self.data.get_content())
            labels = [1 if row[-1] > 0 else 0 for row in self.data.get_content()]
            content = self.split(self.data.get_content(), labels, self.splits)
            for train_inp, train_out, test_inp, test_out in content:
                # Smoting the highly unbalanced class
                if (smote_val):
                    train_inp, train_out = self.balance(train_inp,
                                                    train_out,
                                                    neighbors=5)
                self.data.set_train_data(train_inp)
                self.data.set_train_label(train_out)
                self.data.set_test_data(test_inp)
                self.data.set_test_label(test_out)
                yield self.run()

    def split(self, inp, out, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
        inp, out = np.array(inp), np.array(out)
        for train_index, test_index in skf.split(inp, out):
            yield inp[train_index], out[train_index], inp[test_index], out[test_index]

    def smote(self, data, num, k=5):
        corpus = []
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        for i in range(0, num):
            mid = randint(0, len(data) - 1)
            nn = indices[mid, randint(1, k)]
            datamade = []
            for j in range(0, len(data[mid])):
                gap = random()
                datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
            corpus.append(datamade)
        corpus = np.array(corpus)
        return corpus

    def balance(self, data_train, train_label, neighbors=5):
        pos_train = []
        neg_train = []
        for j, i in enumerate(train_label):
            if i == 1:
                pos_train.append(data_train[j])
            else:
                neg_train.append(data_train[j])
        pos_train = np.array(pos_train)
        neg_train = np.array(neg_train)
        num = int((len(pos_train) + len(neg_train)) / 2)
        if len(pos_train) < len(neg_train):
            pos_train = self.smote(pos_train, num, k=neighbors)
            neg_train = neg_train[np.random.choice(len(neg_train), num, replace=False)]
            data_train1 = np.vstack((pos_train, neg_train))
            label_train = [1] * len(pos_train) + [0] * len(neg_train)
            return data_train1, label_train
        else:
            neg_train = self.smote(neg_train, num, k=neighbors)
            pos_train = pos_train[np.random.choice(len(pos_train), num, replace=False)]
            data_train1 = np.vstack((pos_train, neg_train))
            label_train = [1] * len(pos_train) + [0] * len(neg_train)
            return data_train1, label_train

    def run(self):
        pass
