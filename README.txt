
1. Setup and installation

For this I only used Python 3.6
Install all the packages imported in all files (main packages are scikit-learn, pandas and numpy)



2. Code and running

First step :

preprocessing / preprocess.py, the file is documented, it is just a sequence of operations 
to apply to the initial csv, modify the data, save only the features we want and then save it in a new .csv file

Second step: 

Generating the labels / generate_labels.py, the file is fully documented. You will have to change the list of features that are considered,
the number of desired clusters and the name of the .csv label file that is computed
We need to apply this script several times (~50) with different parameters to compute a large number of labels and make the mean in the step 3

Third step: 

computing the final labels by making the mean of all labels computed for each instance / group_csv.py
The file is documented, it takes all the csv label files and compute the mean of them into a list before saving it again into a final label.csv
Remark : All the names of the .csv are set manually, please change it by hand

Fourth step : 

This is the learning step, it takes the .csv corresponding to the dataset and the .csv corresponding to the labels and try some ML algorithm on it / main_function.py
The script is well documented.
The ML algorithm implemented are DT.py, KNN.py, RandomForest.py, SVM.py (but only for a one class which is bad), Adaboost.py.
Each script contains a "XXXXXXXX_test" function that makes a simple test applying the coresponding algorithm once only and a 
"XXXXXXX_plot" function that does cross validation and generate a graph over the variation on one parameter of the algorithm