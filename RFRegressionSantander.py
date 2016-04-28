#-----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
#import pandas a data processing and CSV file I/O library
import pandas as pd
import numpy as np
# Seaborn -- a python graphing library
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

from VisualizeSantanderData import loadData
#------------------------------------------------------------------------------
def RFScore(trees_in_forest, X, y):
 scores = []
 for trees_count in range(1, trees_in_forest):
  clf = RandomForestClassifier(n_estimators = trees_count)
  '''
  Dividing the training set into cv smaller sets (here cv =10)
  A model(here RF) is trained using cv-1 sets and resulting model is validated
  on the remaining part of the set 
  '''
  validated = cross_val_score(clf, X, y, cv=10)
  scores.append(validated)
 return(scores)
#------------------------------------------------------------------------------
def BOXplot(dt):
 sns.boxplot(data= dt, orient="v")
 plt.xlabel('Number of trees')
 plt.ylabel('Classification scores')
 plt.title('Classification scores vs growing number of trees in forest')
 plt.show()

#-------------------------------------------------------------------------------
def normalize(data, high=1.0, low=0.0):
 mins = np.min(data, axis=0)
 maxs = np.max(data, axis=0)
 range_ = maxs - mins
 return high - (((high - low) * (maxs - data)) / range_)
#-------------------------------------------------------------------------------
def   X_scaled(X):
 preprocessing.scale(X) # axis 0 is default, zero mean and unit var 
#-------------------------------------------------------------------------------
def X_min_max(X):
 # between zero to one
 min_max_scaler = preprocessing.MinMaxScaler()
 return min_max_scaler.fit_transform(X_train)
#-------------------------------------------------------------------------------
def X_normalized(X):
 return preprocessing.normalize(X, norm='l2', axis = 0)
#-------------------------------------------------------------------------------
def executethisModule():

# if __name__ == '__main__':
 
 # NUMBER_ROWS = 9000
 FOREST_TREES_COUNT = 1577 

 #------------------------------------------------------------------------
 # training
 load_csv_train = loadData('train.csv')  # create an instance of loadData
 training_data = load_csv_train.loadCSVFull()
 # training_data = load_csv_train.loadCSVFewRows(NUMBER_ROWS)
 y_target =  training_data['TARGET'].as_matrix()
 X_input  =  training_data.drop(['TARGET'], axis=1).as_matrix()
 # scores  = RFScore(FOREST_TREES_COUNT, X_input, y_target)
 # BOXplot(scores) 
 print("Training started")
 rf = RandomForestClassifier(n_estimators = FOREST_TREES_COUNT)
 
 X_input = X_normalized(X_input)
 
 rf.fit(X_input, y_target)
 
 print("Training finished")
 #-----------------------------------------------------------------------
 # testing
 load_csv_train = loadData('test.csv')  # create an instance of loadData
 testing_data = load_csv_train.loadCSVFull()
 # testing_data = load_csv_train.loadCSVFewRows(NUMBER_ROWS)
 testArr = testing_data.as_matrix()
 customerId =  testArr[:,0]
 
 testArr = X_normalized(testArr) 
 
 predicted_probs = rf.predict_proba(testArr)
 # customerId = customerId.astype(int)
 rows = zip(customerId, predicted_probs[:,1])
 
 # np.savetxt("output.csv", zip(customerId, predicted_probs[:,1]), delimiter=',', header="ID,TARGET", comments="",fmt='%u')
 with open('output_rf.csv', 'wb') as fp:
  csv_writer = csv.writer(fp)
  csv_writer.writerow(('ID', 'TARGET'))
  for row in rows:
   lst = list(row)
   lst[0] = int(lst[0])
   t = tuple(lst)
   csv_writer.writerow(t)

 sns.distplot(predicted_probs[:,0], color="g")
 plt.title(' Customer satisfaction probability distribution')
 plt.show()



