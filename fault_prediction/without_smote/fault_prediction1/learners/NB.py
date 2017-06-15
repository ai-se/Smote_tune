from Handler import Handler
from sklearn.naive_bayes import GaussianNB
import sys

sys.dont_write_bytecode = True


class NB(Handler):
    """docstring for NB"""
    def __init__(self, *args):
        super(NB, self).__init__(*args)

    def __str__(self):
        return "Naive Bayes"

    def run(self):
        NB = GaussianNB()
        NB.fit(self.data.get_train_data(), self.data.get_train_label())
        return NB.predict(self.data.get_test_data())
