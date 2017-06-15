from Handler import Handler
from sklearn.ensemble import RandomForestClassifier
import sys

sys.dont_write_bytecode = True


class RF(Handler):
    """docstring for RF"""
    def __init__(self, *args):
        super(RF, self).__init__(*args)

    def __str__(self):
        return "Random Forest"

    def run(self):
        RF = RandomForestClassifier(criterion='entropy')
        RF.fit(self.data.get_train_data(), self.data.get_train_label())
        return RF.predict(self.data.get_test_data())
