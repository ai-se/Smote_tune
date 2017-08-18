from Handler import Handler
from sklearn.linear_model import LogisticRegression
import sys

sys.dont_write_bytecode = True


class LR(Handler):
    """docstring for LR"""
    def __init__(self, *args):
        super(LR, self).__init__(*args)

    def __str__(self):
        return "Linear Regression"

    def run(self):
        LR = LogisticRegression()
        LR.fit(self.data.get_train_data(), self.data.get_train_label())
        return LR.predict(self.data.get_test_data())