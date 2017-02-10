#!/usr/bin/env python
from __future__ import print_function

from config import LEARNERS
from random import seed
from Data import Data
from Result import Result
from ReadFile import ReadFile
from stats.sk import rdivDemo
from stats.ABCD import ABCD
import numpy as np
import sys

sys.dont_write_bytecode = True


class Learner(object):
    """
        1) This class is the entry point to execute learners.
        2) :params list_of_learners --> provide a list of learners.
           :params folds --> Data object.
           :params result --> Result object.
        3) Learner names can be found in config.py
    """
    def __init__(self, samples=[],labels=[], smote=True,v=[],percentage=20):
        super(Learner, self).__init__()
        self.samples = samples
        self.labels = labels
        self.smote_val=smote
        self.result = Result()
        self.predict = None
        self.data=Data()
        self.l=v
        self.per=percentage

    @staticmethod
    def show_available_learners():
        return ", ".join([k for k in LEARNERS])

    def run(self, learners=[k for k in LEARNERS], round_results=3):
        for learner_name in learners:
            learner = LEARNERS[learner_name](self.data, self.result)
            split = split_two(np.array(self.samples), np.array(self.labels))
            pos = np.array(split['pos'])
            neg = np.array(split['neg'])
            cut_pos, cut_neg = cut_position(pos, neg, percentage=self.per)
            data_train, train_label, data_test, test_label = divide_train_test(pos, neg, cut_pos, cut_neg)
            data_train, train_label = learner.execute(self.l, samples=data_train, labels=train_label)
            self.data.set_train_data(data_train)
            self.data.set_test_data(data_test)
            self.data.set_train_label(train_label)
            self.data.set_test_label(test_label)
            self.predict = [learner.run()]
            for prediction in self.predict:
                def generate_stats(predict):
                    abcd = ABCD(before=self.data.get_test_label(), after=predict)
                    stats = np.array([j.stats() for j in abcd()])
                    labels = list(set(self.data.get_test_label()))
                    if labels[0] == 0:
                        target_label = 1
                    else:
                        target_label = 0
                    r_val = stats[target_label][0]
                    p_val = stats[target_label][3]
                    a_val = stats[target_label][4]
                    f_score_val = stats[target_label][5]
                    pf_val=stats[target_label][1]
                    return r_val, p_val, a_val, f_score_val,pf_val

                recall, precision, accuracy, f_score,pf = generate_stats(prediction)
                self.result.set_recall(learner_name, round(recall, round_results))
                self.result.set_precision(learner_name, round(precision, round_results))
                self.result.set_accuracy(learner_name, round(accuracy, round_results))
                self.result.set_f_score(learner_name, round(f_score, round_results))
                self.result.set_false_alarm(learner_name, round(pf, round_results))

    def get_recall(self):
        return self.result.get_recall()

    def get_precision(self):
        return self.result.get_precision()

    def get_accuracy(self):
        return self.result.get_accuracy()

    def get_f_score(self):
        return self.result.get_f_score()

    def get_false_alarm(self):
        return self.result.get_false_alarm()

    def display_stats(self):
        for k, v in self.result.scores.items():
            print(k)
            rdivDemo(v())
            print("")

def cut_position(pos, neg, percentage=0):
    return int(len(pos) * percentage / 100), int(len(neg) * percentage / 100)

def divide_train_test(pos, neg, cut_pos, cut_neg):
    data_train, train_label = list(pos)[:cut_pos] + list(neg)[:cut_neg], [1] * (cut_pos) + [0] * (cut_neg)
    data_test, test_label = list(pos)[cut_pos:] + list(neg)[cut_neg:], [1] * (len(pos) - cut_pos) + [0] * (
        len(neg) - cut_neg)
    return np.array(data_train), train_label, np.array(data_test), test_label

def split_two(corpus=np.array([]), label=np.array([])):
    pos = []
    neg = []
    for i, lab in enumerate(label):
        if lab == 1.0:
            pos.append(corpus[i])
        else:
            neg.append(corpus[i])

    return {'pos': pos, 'neg': neg}