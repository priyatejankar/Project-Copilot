import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from io import StringIO
from IPython.display import Image
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve


# values = np.genfromtxt('old_ml_sup_proect/dataset_eye.csv', delimiter = ',', dtype = (float))
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
# y_train_values = values[indices[:-1500],-1:]
# mel_y_train_values = values[indices,-1:]
# #y_train_values = values_w[indices[:-1500],-1:]
# test_values = norm_values[indices[-1500:]]
# y_test_values = values[indices[-1500:],-1:]
# #y_test_values = values_w[indices[-1500:],-1:]
# #print(train_values)

def Random_Forest_test( train_set, label_train,  validation_set, label_validation, depth =8):
    print("Decision trees test")
    clf = ensemble.RandomForestRegressor(n_estimators = 15, criterion = 'mse', max_depth=depth)
    clf = clf.fit(train_set, label_train)
    trees_computed_values = clf.predict(validation_set)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(label_validation, trees_computed_values, multioutput='uniform_average')
    print("mean error is : ", mean_absolute_error)
    mean_max_absolute_error_arrays = np.mean(np.amax(np.absolute(np.subtract(trees_computed_values, label_validation)), axis = 1))
    print("mean max error per row is : ", mean_max_absolute_error_arrays)
    return clf

def Random_Forest_plot( train_set, label_train,  validation_set, label_validation, depth =8):
    print("Decision trees test")
    clf = ensemble.RandomForestRegressor(n_estimators=15, criterion='mse', max_depth=depth)
    # label_train = np.squeeze(label_train)
    clf = clf.fit(train_set, label_train)
    trees_computed_values = clf.predict(validation_set)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(label_validation, trees_computed_values,
                                                              multioutput='uniform_average')
    print("mean error is : ", mean_absolute_error)
    print(trees_computed_values)
    # trees_computed_values = np.array([[0.5, 0.5, 1], [0.7, 0.7,2]])
    # label_validation = np.array([[0.5, 0.53, 1], [0.75, 0.7, 2]])
    mean_max_absolute_error_arrays = np.mean(
        np.amax(np.absolute(np.subtract(trees_computed_values, label_validation)), axis=1))
    print("mean max error per row is : ", mean_max_absolute_error_arrays)

    X = np.concatenate((train_set, validation_set), axis=0)
    # label_validation = np.squeeze(label_validation)
    y = np.concatenate((label_train, label_validation), axis=0)
    # z = np.transpose(norm_values[:, -1:])[0]
    param_range = np.arange(3, 35, 2)
    train_scores, test_scores = validation_curve(ensemble.RandomForestRegressor(n_estimators=35, criterion='mse'), X, y,
                                                 param_name="max_depth", param_range=param_range, cv=6,
                                                 scoring="neg_mean_squared_error", n_jobs=4)
    train_scores_mean = np.mean(-train_scores, axis=1)*8
    train_scores_std = np.std(-train_scores, axis=1)*8
    test_scores_mean = np.mean(-test_scores, axis=1)*8
    test_scores_std = np.std(-test_scores, axis=1)*8
    # print(test_scores)
    plt.title("RandomForest, Toy dataset (200 000 RTA), n_estimators = 35")
    plt.xlabel("max_depth")
    plt.ylabel("Total mean squared error")
    plt.ylim(0.0, 0.2)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()



# Random_Forest_plot( mel_train_values, mel_y_train_values,  test_values, y_test_values)






