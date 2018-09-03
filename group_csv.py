from SVM import *
from DT import *
from KNN import *
from RandomForest import *
from Adaboost import *
from clustering.EM import *
from clustering.KMeans import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import random
import numpy as np
import csv

concatenated_labels = []
# with open('Labels_rep/labels_8_1.csv', 'r') as f:
#   reader = csv.reader(f)
#   list1 = list(reader)
# with open('Labels_rep/labels_8_2.csv', 'r') as f:
#   reader = csv.reader(f)
#   list2 = list(reader)

# Use of an iterator to avoid memory errors
# select the first csv file containing labels
targets = ((float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h))
           for a, b, c, d, e, f, g, h in csv.reader(open('Labels_rep/labels_8_1.csv','r')))
# print(len(targets))
i = 0
length_features = 8
# put the first list of labels in the final list
for target in targets:
    # select only the 200 000 first
    # TODO try it with the whole dataset
    if i == 200000:
        break
    i += 1
    concatenated_labels.append(list(target))

# sum all other csv label files to the first one (line by line on the list
for index in range(8,18):
    if not index == 8:
        targets = ((float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h))
                   for a, b, c, d, e, f, g, h in csv.reader(open('Labels_rep/labels_'+ str(index) +'_1.csv', 'r')))
        i=0
        for target in targets:
            # select only the 200 000 first
            # TODO try it with the whole dataset
            if i == 200000:
                break
            for jindex in range(length_features):
                concatenated_labels[i][jindex] += target[jindex]
            i += 1
    targets = ((float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h))
               for a, b, c, d, e, f, g, h in csv.reader(open('Labels_rep/labels_'+ str(index) +'_2.csv', 'r')))
    i = 0
    for target in targets:
        # select only the 200 000 first
        # TODO try it with the whole dataset
        if i == 200000:
            break
        for jindex in range(length_features):
            concatenated_labels[i][jindex] += target[jindex]
        i += 1

# divide each term by the total number of label files to get the mean
# TODO delete the magic number /20

for iindex in range(len(concatenated_labels)):
    for jindex in range(length_features):
        concatenated_labels[iindex][jindex] = concatenated_labels[iindex][jindex] /20.0
print(len(concatenated_labels))
labels = np.array(concatenated_labels)

# save the final labels into a csv
np.savetxt('Labels_rep/labels_final_200000.csv', labels, delimiter=',')


















# (length_i, length_j) = concatenated_labels[0].shape
# number_of_runs = len(concatenated_labels)
# float_number_of_runs = float(number_of_runs)
# labels = concatenated_labels[0]
# for i in range(length_i):
#     for j in range(length_j):
#         aux = 0
#         for k in range(number_of_runs):
#             aux += concatenated_labels[k][i,j]
#         labels[i,j] = aux / float_number_of_runs
# np.savetxt('Labels_rep/labels_final.csv', labels, delimiter=',')


# df1 = pd.read_csv("Labels_rep/labels_8_1.csv", names = [0,1,2,3,4,5,6,7])
# df2 = pd.read_csv("Labels_rep/labels_8_2.csv", names = [0,1,2,3,4,5,6,7])

# df = pd.concat((df1, df2), axis = 1)
# print(df.shape)
# df.stack().groupby(level=[0]).mean().unstack()
# print(df)
# for i in range(8, 18):
#     df1 = pd.read_csv('Labels_rep/labels_'+ str(i) +'_1.csv', names=[0, 1, 2, 3, 4, 5, 6, 7])
#     df2 = pd.read_csv('Labels_rep/labels_'+ str(i) +'_2.csv', names=[0, 1, 2, 3, 4, 5, 6, 7])
#     print(i)
#     # print(my_data_1)
#     concatenated_labels.append(np.genfromtxt('Labels_rep/labels_'+ str(i) +'_1.csv', delimiter=','))
# for i in range(8, 18):
#     my_data_2 = np.genfromtxt('Labels_rep/labels_'+ str(i) +'_2.csv', delimiter=',')
#     print(i)
#     concatenated_labels.append(my_data_2)

#

#
