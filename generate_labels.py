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

# names of the columns of the csv
column_names = ['Longitude', 'Latitude', 'Time', 'Speed_limit',
                'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area']

# read the csv
df = pd.read_csv("Reduced_accidents0515.csv", names=column_names)
# drop empty rows
df=df.dropna()
# drop the two first rows that are usually unadapted (column names, etc)
df.drop(0, inplace=True)
df.drop(1, inplace=True)
# transform it into a numpy array
full_dataset = df.as_matrix()
# print("coucou1")

# The .csv file is not read as integers / float, especially the latitude and longitude columns
# there is a need to change the coma to a dot that allow us to convert a float number written as a
# string to a float without trouble
for index in range(full_dataset.shape[0]):
    if type(full_dataset[index][0]) == str :
        full_dataset[index][0] = full_dataset[index][0].replace(',', '.')
    if type(full_dataset[index][1]) == str:
        full_dataset[index][1] = full_dataset[index][1].replace(',', '.')

# cast the whole numpy array as a float64 numpy array
full_dataset = full_dataset.astype(np.float64)
# check if there is NaNs in the array
where_are_nans =  np.isnan(full_dataset)
# print(np.argwhere(where_are_nans==True))
# print("coucou4")
# print(full_dataset)
# print(full_dataset.shape)
# print(type(full_dataset[0][0]))
# print(full_dataset.shape)
# full_dataset = full_dataset[:200000][:]
# print(np.isnan(full_dataset).any())

concatenated_labels = []

# First need to label the data, apply 50 cluster algorithm compute a label for each cluster and
# assign it to each instance in that cluster.

# Need of a '__main__' function for the parallel computing on windows
if __name__ == '__main__':

    # NEED THE STANDARD SCALER TO NORMALIZE THE DATA
    # We will need to save the Scaler later to be able to transform input features,
    # that means the features computed in real time for the user in the same way
    # TODO save the scaler
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(full_dataset)
    full_dataset = scaler.transform(full_dataset)
    print(full_dataset.shape)

    # for cluster_index in range(3):
    #     for repeat_index in range(1):
    #         print(" (cluster_index, repeat_index) : ", cluster_index, repeat_index)
    #         cluster_vector = np.copy(apply_Kmeans(full_dataset, number_clusters=7 + 1*cluster_index))
    #         concatenated_labels.append(np.copy(apply_PCA(full_dataset, cluster_vector)))
    #         print(len(concatenated_labels))

    # print(" (cluster_index, repeat_index) : ", cluster_index, repeat_index)

    # TODO find a way to compute several labels vectors in the same script without memory errors
    # Apply Kmeans to compute the clusters, use of np.copy to do a deep copy of the vector and avoid cross references
    cluster_vector = np.copy(apply_Kmeans(full_dataset, number_clusters=17))
    # apply PCA and compute the label for each instance in the dataset
    concatenated_labels.append(np.copy(apply_PCA(full_dataset, cluster_vector)))

    # transform the labels into a numpy array, thispart will be usefull when there will be several label arrays
    number_of_runs = len(concatenated_labels)
    float_number_of_runs = float(number_of_runs)
    (length_i, length_j) = concatenated_labels[0].shape
    labels = np.array(concatenated_labels[0])
    # print(concatenated_labels[0][0])
    # print(concatenated_labels[0][1])
    # print(concatenated_labels[0][2])
    #
    # print("np_concatenated_labels")
    # print(labels)
    print("np_concatenated_labels.shape")
    print(labels.shape)

    # Then compute the mean of all labels to find an "approximation" of the risks
    # labels = np.mean(np_concatenated_labels, axis=0)

    # Save the labels in a .csv, we will join all the labelXXXX.csv in another script
    np.savetxt('Labels_rep/labels_17_2.csv', labels, delimiter=',')

