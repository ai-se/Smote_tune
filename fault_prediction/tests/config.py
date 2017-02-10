import NB, KNN, DT, SVM, RF, LR
import sys

sys.dont_write_bytecode = True


LEARNERS = {
    'NB': NB.NB,
    'KNN': KNN.KNN,
    'DT': DT.DT,
    'LR': LR.LR,
    'SVM': SVM.SVM,
    'RF': RF.RF
}