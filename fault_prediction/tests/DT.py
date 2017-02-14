from sklearn.tree import DecisionTreeClassifier
import sys
from fault_prediction.stats.ABCD import ABCD
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.dont_write_bytecode = True


class DT(object):
    """docstring for DT"""
    def __init__(self, *args):
        pass

    def run(self,data_train, train_label, data_test, test_label):
        DT = DecisionTreeClassifier(criterion='entropy')
        DT.fit(data_train, train_label)
        prediction=DT.predict(data_test)
        abcd = ABCD(before=test_label, after=prediction)
        stats = np.array([j.stats() for j in abcd()])
        labels = list(set(test_label))
        if labels[0] == 0:
            target_label = 1
        else:
            target_label = 0
        #fpr, tpr, _ = roc_curve(test_label, prediction, pos_label=target_label)
        #auc1 = auc(fpr, tpr)
        return stats[target_label][1]
        #return auc1
