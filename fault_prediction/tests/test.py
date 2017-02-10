from fault_prediction.Learner import Learner
from fault_prediction.stats.demos import *
from ReadFile import ReadFile
import sys
from random import *
import time
import copy
import collections
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from smote import *

sys.dont_write_bytecode = True
learners=['NB', 'KNN', 'DT','LR', 'SVM', 'RF']

files = ['tomcat', 'xalan','synapse', 'xerces', 'camel', 'prop', 'ant', 'arc', 'poi', 'ivy', 'velocity', 'redaktor', 'log4j', 'jedit']

#m, r, k , e
bounds = [[50,100,200,400], list(range(1, 6)), list(range(5, 21)), (0.3,3)]

__all__ = ['DE']
Individual = collections.namedtuple('Individual', 'ind fit1 fit2')


class DE(object):
    def __init__(self, x='rand', y=1, z='bin', F=0.3, CR=0.7):
        self.x = x
        self.y = y
        self.z = z
        self.F = F
        self.CR = CR

    def solve(self, fitness, initial_population, iterations=10, **r):
        current_generation=[]
        for ind in initial_population:
            a,b=fitness(*ind, **r)
            current_generation.append(Individual(ind, a,b))

        l=[]
        for i in current_generation:
            l.append([i.ind,i.fit1,i.fit2])
        for _ in range(iterations):
            trial_generation = []

            for ind in current_generation:
                v = self._extrapolate(ind,current_generation)
                a1,b1=fitness(*v, **r)
                trial_generation.append(Individual(v, a1,b1))
                l.append([v, a1,b1])

            current_generation = self._selection(current_generation,
                                                 trial_generation)

        best_index = self._get_best_index(current_generation)
        return current_generation[best_index].ind, current_generation[best_index].fit1, l

    def select3others(self,population):
        popu=copy.deepcopy(population)
        x= randint(0, len(popu)-1)
        x1=popu[x]
        popu.pop(x)
        y= randint(0, len(popu)-1)
        y1=popu[y]
        popu.pop(y)
        z= randint(0, len(popu)-1)
        z1=popu[z]
        popu.pop(z)
        return x1.ind,y1.ind,z1.ind

    def _extrapolate(self, ind, population):
        if (random() < self.CR):
            x,y,z=self.select3others(population)
            #print(x,y,z)
            mutated=[choice(bounds[0]), choice(bounds[1]),choice(bounds[2]), x[3] + self.F*(y[3] - z[3])]

            check_mutated= [mutated[0],mutated[1] ,mutated[2],max(bounds[3][0], min(mutated[3], bounds[3][1]))]
            return check_mutated
        else:
            return ind.ind

    def _selection(self, current_generation, trial_generation):
        generation = []

        for a, b in zip(current_generation, trial_generation):
            if (a.fit1+np.median(a.fit2)) >= (b.fit1+np.median(b.fit2)):
                generation.append(a)
            else:
                generation.append(b)

        return generation

    def _get_indices(self, n, upto, but=None):
        candidates = list(range(upto))

        if but is not None:
            # yeah O(n) but random.sample cannot use a set
            candidates.remove(but)

        return sample(candidates, n)

    def _get_best_index(self, population):
        global max_fitness
        best = 0

        for i, x in enumerate(population):
            if (x.fit1+np.median(x.fit2)) >= max_fitness:
                best = i
                max_fitness = x.fit1+np.median(x.fit2)
        return best

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

def _test(res=''):

    start_time = time.time()
    filepath = '../data/'+res+'.csv'

    seed(1)
    read=ReadFile(filepath)
    data,labels=read.build_table()

    ## Normalize Preprocessing
    #data = normalize(np.array(zip(*data)), norm='l2')
    #data=np.array(zip(*data))

    split=split_two(np.array(data),np.array(labels))
    pos = np.array(split['pos'])
    neg = np.array(split['neg'])

    ## 20% train and test
    final={}
    result={}
    cut_pos, cut_neg = cut_position(pos, neg, percentage=80)
    for learner in learners:
        start_time1 = time.time()
        l=[]
        x = {}
        measures = ["Recall", "Precision", "Accuracy", "F_score", "False_alarm", "AUC"]
        for q in measures:
            x[q]=[]
        for folds in range(15):
            pos_shuffle = range(0, len(pos))
            neg_shuffle = range(0, len(neg))
            shuffle(pos_shuffle)
            shuffle(neg_shuffle)
            pos = pos[pos_shuffle]
            neg = neg[neg_shuffle]
            data_train, train_label, data_test, test_label = divide_train_test(pos, neg, cut_pos, cut_neg)
            de = DE(F=0.7, CR=0.3, x='rand')

            global max_fitness
            max_fitness = 0
            pop = [[choice(bounds[0]), choice(bounds[1]),
                    choice(bounds[2]), uniform(bounds[3][0], bounds[3][1])]
                   for _ in range(10)]

            v, score, final_para_dic = de.solve(main, pop, iterations=3, file=res,
                                                data_samples=data_train, target=train_label, learner=learner)

            model = Learner(samples=np.vstack((data_train,data_test)),labels=train_label+test_label,
                              smote=True,v=v,percentage=80)
            model.run(learners=[learner])
            x["Accuracy"].append(model.get_accuracy()[0][1])
            x["F_score"].append(model.get_f_score()[0][1])
            x["Precision"].append(model.get_precision()[0][1])
            x["Recall"].append(model.get_recall()[0][1])
            x["False_alarm"].append(model.get_false_alarm()[0][1])
            x["AUC"].append(model.get_auc()[0][1])
            l.append([v, score, final_para_dic])
        print(x)
        result[learner] = [x,l,time.time()-start_time1]
    final[res]=result
    print(final)
    with open('../dump/' + res + '.pickle', 'wb') as handle:
        pickle.dump(final, handle)

if __name__ == '__main__':
    eval(cmd())
