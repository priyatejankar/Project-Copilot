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

def SVM_test( train_set, label_train,  validation_set, label_validation, depth =8):
    print("SVM test")
    clf = svm.SVR(C=1.0, gamma='auto', kernel='rbf', degree=3, tol=0.0001)
    clf.fit(train_set, label_train)
    svm_computed_values = clf.predict(validation_set)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(label_validation, svm_computed_values, multioutput='uniform_average')
    print("mean error is : ", mean_absolute_error)
    mean_max_absolute_error_arrays = np.mean(np.amax(np.absolute(np.subtract(svm_computed_values, label_validation)), axis = 1))
    print("mean max error per row is : ", mean_max_absolute_error_arrays)
    return clf

def SVM_plot( train_set, label_train,  validation_set, label_validation, depth =8):
    print("SVM test")
    # label_train = np.squeeze(label_train)
    # label_validation = np.squeeze(label_validation)
    clf = svm.SVR(C=1.0, gamma='auto', kernel='rbf', degree=3, tol=0.0001)
    clf.fit(train_set, label_train)
    svm_computed_values = clf.predict(validation_set)
    mean_absolute_error = sklearn.metrics.mean_absolute_error(label_validation, svm_computed_values,
                                                              multioutput='uniform_average')
    print("mean error is : ", mean_absolute_error)
    mean_max_absolute_error_arrays = np.mean(
        np.amax(np.absolute(np.subtract(svm_computed_values, label_validation)), axis=1))
    print("mean max error per row is : ", mean_max_absolute_error_arrays)

    X = np.concatenate((train_set, validation_set), axis=0)
    y = np.concatenate((label_train, label_validation), axis=0)

    # z = np.transpose(norm_values[:, -1:])[0]

    param_range = np.logspace(-3, 2, 10)
    train_scores, test_scores = validation_curve(
        svm.SVR(gamma='auto', kernel='rbf', degree=3, tol=0.0001), X, y,
        param_name="C", param_range=param_range,
        scoring="neg_mean_absolute_error", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print(test_scores)
    plt.title("Eye SVM : Kernel 'rbf'")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(-10, 10)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


# SVM_plot( mel_train_values, mel_y_train_values,  test_values, y_test_values)




