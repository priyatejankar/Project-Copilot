# for the pca https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn import datasets


from sklearn import preprocessing

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import sklearn.metrics
from sklearn.datasets import load_digits
#from sklearn.model_selection import validation_curve


# values = np.genfromtxt('C:/Users/BUCHET PIERRE/Desktop/Copilot_project/old_ml_unsup_project/dataset_eye2.csv', delimiter = ',', dtype = (float))
#
# #norm_values = values[:,:-1] / np.linalg.norm(values[:,:-1]) * values.shape[0]
# scaler = preprocessing.StandardScaler().fit(values[:,:-1])
# print(scaler)
# norm_values = scaler.transform(values[:,:-1])
# #norm_values = scaler.transform(values_w[:,:-1])
# print(norm_values.shape)
# indices = np.random.permutation(values.shape[0])
# #indices = np.random.permutation(values_w.shape[0])
# train_values =norm_values[indices[:-1500]]
# mel_train_values =norm_values[indices]
# mel_y_train_values = values[indices,-1:]
# y_train_values = values[indices[:-1500],-1:]
# #y_train_values = values_w[indices[:-1500],-1:]
# test_values = norm_values[indices[-1500:]]
# y_test_values = values[indices[-1500:],-1:]
# #y_test_values = values_w[indices[-1500:],-1:]
# print(train_values)


def apply_Kmeans(full_dataset, number_clusters = 8):
    cltr = KMeans(n_clusters=number_clusters, init='random', n_init=1, max_iter=100, tol=0.001, precompute_distances=False,
                        verbose=0, random_state=None, copy_x=True, algorithm='full',  n_jobs=4)
    cltr = cltr.fit(full_dataset)
    # print(cltr.cluster_centers_)
    vector_cluster = cltr.predict(full_dataset)
    print("Fin Kmeans")
    return vector_cluster


def apply_PCA(full_dataset, vector_cluster):
    num_clusters = np.max(vector_cluster) + 1
    print("num cluster : ", num_clusters)
    new_labels = full_dataset
    for i in range(num_clusters):
        cluster_data = full_dataset[vector_cluster == i,:]
        # print(cluster_data.shape)
        pca_cluster = PCA(n_components=1, svd_solver = 'full')
        if cluster_data.shape[0] == 0:
            print("problem : shape is ", cluster_data.shape)
            continue
        pca_cluster.fit(cluster_data)
        pca_vector = pca_cluster.components_[0]
        for index in range(pca_vector.shape[0]):
            pca_vector[index] = max(abs(pca_vector[index]), 0.0001)
        inv_pca_vector = np.reciprocal(pca_vector)
        norm = np.linalg.norm(inv_pca_vector)
        normalized_label = inv_pca_vector / norm
        # print(normalized_label)
        new_labels[vector_cluster == i] = normalized_label
        # print(new_labels[i])
        # print(vector_cluster)
        # print(vector_cluster == i)
        # print(pca_cluster.components_)
        # print(pca_cluster.explained_variance_)
    print("Fin PCA")
    return new_labels

# vector_print = apply_Kmeans(mel_train_values)
# print(vector_print)
# print(np.min(vector_print))
# print(np.max(vector_print))
# return_labels = apply_PCA(mel_train_values, vector_print)
# print(return_labels)