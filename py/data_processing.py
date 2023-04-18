# Importing libraries
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#Evaluation tools
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
##
from sklearn.model_selection import train_test_split
#CV
from sklearn.model_selection import cross_val_score
#TODO: Download Dublin bike dataset 'Dublinbikes 2021 Q1 usage data' from https://data.gov.ie/dataset/dublinbikes-api and save it in the same location as this jupyter notebook
##
#data_path = "./data"
data_path = "C:/Data/tcd_asds/data_files/mlData/"
# read all data files in directory
#file_name = "fruitDataset.csv"
#file_name = "C:/Data/tcd_asds/data_files/mlData/dublinbikes_20210101_20210401.csv"
file_name = "dublinbikes_20210101_20210401.csv"
#dataset = pd.read_csv("dublinbikes_20210101_20210401.csv")
dataset = pd.read_csv(data_path + file_name)
##
# get file names
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
##
print(dataset.head())
dataset.columns

nbr_col = len(dataset.columns)
print('There are', nbr_col, 'columns.')

dataset.hist(bins = 50, figsize = (20, 15))
plt.show()
##
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
##
print(train_set.columns)
##
train_set['BIKE STANDS'].hist()
plt.show()
##
train_set['bikes_avail_cat'] = pd.cut(train_set['BIKE STANDS'], bins = [0.,1.5, 3.0, 4.5, 6., np.inf], \
                              labels = [1,2,3,4,5])
housing['income_cat'].hist()

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
##
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set['income_cat'].value_counts() / len(strat_test_set)
##
mask_pearse = dataset.NAME == 'PEARSE STREET'
pearse_data = dataset[mask_pearse]
print(pearse_data)


selected_time_indices = dataset.TIME == '2021-03-02 14:00:03'
dataset[selected_time_indices]

selected_time_data = dataset[selected_time_indices]
mask_good_stations = selected_time_data['AVAILABLE BIKE STANDS']>15
good_stations = selected_time_data.NAME[mask_good_stations]
print(np.unique(good_stations))

##

test_
