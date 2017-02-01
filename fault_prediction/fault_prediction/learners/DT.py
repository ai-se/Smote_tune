from Handler import Handler
from sklearn.tree import DecisionTreeClassifier
import sys

sys.dont_write_bytecode = True


class DT(Handler):
    """docstring for DT"""
    def __init__(self, *args):
        super(DT, self).__init__(*args)

    def __str__(self):
        return "Decision Tree"

    def run(self):
        DT = DecisionTreeClassifier(criterion='entropy')
        DT.fit(self.data.get_train_data(), self.data.get_train_label())
        return DT.predict(self.data.get_test_data())
