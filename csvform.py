__author__ = 'amrit'

import matplotlib.pyplot as plt
import os, pickle
import numpy as np
import csv

if __name__ == '__main__':

    fileB = []
    F_final1 = {}
    current_dic1 = {}
    para_dict1 = {}
    time1 = {}
    path = '/Users/amrit/GITHUB/fss16591/project/fault_prediction/dump/new/without/'
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            a = os.path.join(root, name)
            with open(a, 'rb') as handle:
                F_final = pickle.load(handle)
                F_final1 = dict(F_final1.items() + F_final.items())
    #print(F_final1)


    measure_med = {}
    measure_iqr = {}
    l = ["Recall", "Precision", "Accuracy", "F_score","False_alarm"]
    for i in l:
        measure_med[i] = {}
        measure_iqr[i] = {}
    for f, measures in F_final1.iteritems():
        fileB.append(f)
        for mea, values in measures.iteritems():
            for k in values:
                try:
                    measure_med[mea][k[0]].append(np.median(k[1:]))
                    measure_iqr[mea][k[0]].append(np.percentile(k[1:], 75) - np.percentile(k[1:], 25))
                except KeyError:
                    measure_med[mea][k[0]] = [np.median(k[1:])]
                    measure_iqr[mea][k[0]] = [np.percentile(k[1:], 75) - np.percentile(k[1:], 25)]
    X = range(len(fileB))

    with open('fault_prediction/dump/new/nosmote.csv', 'a+') as csvinput:
        fields = ['Learners', 'Measures']+['synapse', 'xerces', 'tomcat', 'xalan', 'camel', 'prop', 'ant', 'arc', 'poi', 'ivy', 'velocity', 'redaktor', 'log4j', 'jedit']

        writer = csv.writer(csvinput, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fields)

        for i,j in enumerate(measure_iqr.keys()):
            plt.figure(num=i, figsize=(25, 15))
            for k in measure_iqr[j].keys():
                l=[]
                for x,y in enumerate(measure_med[j][k]):
                    l.append("{0:.2f}".format(measure_med[j][k][x]) + ' / ' + "{0:.2f}".format(measure_iqr[j][k][x]))

                writer.writerow([k,  j]+l)

