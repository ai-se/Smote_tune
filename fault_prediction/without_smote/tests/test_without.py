from fault_prediction1 import Learner
from fault_prediction1.stats.demos import *
import sys
import pickle
from random import seed

sys.dont_write_bytecode = True
# import os
# path=os.getcwd()+'/../'
# print(path)
# sys.path.insert(1,path)

fil = ['tomcat', 'xalan','synapse', 'xerces', 'camel', 'prop', 'ant', 'arc', 'poi', 'ivy', 'velocity', 'redaktor', 'log4j', 'jedit']

def _test(res=''):
    file = res
    seed(1)
    learner = Learner('../data/' + file + '.csv',folds=3,splits=5, smote=False)
    learner.run()
    result = {}
    x = {}
    x["Accuracy"] = learner.get_accuracy()
    x["F_score"] = learner.get_f_score()
    x["Precision"] = learner.get_precision()
    x["Recall"] = learner.get_recall()
    x["False_alarm"] = learner.get_false_alarm()
    x["AUC"] = learner.get_false_alarm()
    result[file] = x
    print(x)
    with open('../dump/new/without/' + file + '.pickle', 'wb') as handle:
        pickle.dump(result, handle)

if __name__ == '__main__':
    eval(cmd())
