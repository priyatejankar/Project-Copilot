from SVM import *
from DT import *
from KNN import *
from RandomForest import *
from Adaboost import *
from clustering.EM import *
from clustering.KMeans import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random


if __name__ == '__main__':

    # names of the columns of the csv
    column_names = ['Longitude', 'Latitude', 'Time', 'Speed_limit',
                    'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area']
    # read it as a panda dataframe
    df = pd.read_csv("Reduced_accidents0515.csv", names=column_names)
    # remove empty rows
    df=df.dropna()
    #remove headers
    df.drop(0, inplace=True)
    df.drop(1, inplace=True)

    # transform it into a numpy array
    full_dataset = df.as_matrix()

    # Again apply a scaler (this one may not be useful)
    # TODO try without it
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(full_dataset)
    full_dataset = scaler.transform(full_dataset)
    full_dataset = full_dataset[:200000][:]

    # test duplicates
    # unique_list = []
    # for index in range(full_dataset.shape[0]):
    #     if not(list(full_dataset[index]) in unique_list):
    #         unique_list.append(list(full_dataset[index]))

    # read the labels
    labels = np.genfromtxt('Labels_rep/labels_final_200000.csv', delimiter=',')

    # print(labels.shape)
    # indices = np.random.permutation(labels.shape[0])
    # full_dataset, labels = shuffle(full_dataset, labels)

    # Separate training / validation set, the last commentend lign of this paragraph may be better,
    # TODO use the train_test_split function but still troubles with the shuffle
    full_train_dataset = full_dataset[:150000][:]
    full_test_dataset = full_dataset[150000:][:]
    labels_train_dataset = labels[:150000][:]
    labels_test_dataset = labels[150000:][:]
    # full_train_dataset, full_test_dataset, labels_train_dataset, labels_test_dataset = train_test_split(full_dataset, labels, test_size=0.30, shuffle=True)

    # Apply a learning algorithm
    # _plot function does a bunch of stuff using cross validation and different parameters to compute a graph of the
    # corresponding ML algorithm
    # _test function is a test with the algorithm applied once
    clf = Decision_trees_plot(full_train_dataset, labels_train_dataset, full_test_dataset, labels_test_dataset)
    # clf = Random_Forest_plot(full_train_dataset, labels_train_dataset, full_test_dataset, labels_test_dataset)
    # clf = SVM_test(full_train_dataset, labels_train_dataset, full_test_dataset, labels_test_dataset)
    # clf = KNN_plot(full_train_dataset, labels_train_dataset, full_test_dataset, labels_test_dataset)
