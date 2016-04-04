'''
-> Definition (Source - Kaggle):

A random forest is an ensemble of decision trees which will output a prediction value, in this case survival.
Each decision tree is constructed by using a random subset of the training data.
After you have trained your forest, you can then pass each test row through it, in order to output a prediction.
Simple! Well not quite!

Random forest is solid choice for nearly any prediction problem (even non-linear ones)
It's a relatively new machine learning strategy (it came out of Bell Labs in the 90s)
and it can be used for just about anything.
It belongs to a larger class of machine learning algorithms called ensemble methods

-> Ensemble Learning

Ensemble learning involves the combination of several models to solve a 
single prediction problem.
It works by generating multiple classifiers/models which learn and make predictions independently. 
Those predictions are then combined into a single (mega) prediction that should be as good or 
better than the prediction made by any one classifer.
One of the best use cases for random forest is feature selection

'''
#-----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
#import pandas a data processing and CSV file I/O library
import pandas as pd
import numpy as np
# Seaborn -- a python graphing library
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score


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
 plt.title('Classification scores for number of trees')
 plt.show()

#-------------------------------------------------------------------------------
 
if __name__ == '__main__':
 
 NUMBER_ROWS = 1000
 FOREST_TREES_COUNT = 10 

 #------------------------------------------------------------------------
 # training
 load_csv_train = loadData('train.csv')  # create an instance of loadData
 # training_data = load_csv_train.loadCSVFull()
 training_data = load_csv_train.loadCSVFewRows(NUMBER_ROWS)
 y_target =  training_data['TARGET'].as_matrix()
 X_input  =  training_data.drop(['TARGET'], axis=1).as_matrix()
 scores  = RFScore(FOREST_TREES_COUNT, X_input, y_target)
 BOXplot(scores) 
 rf = RandomForestClassifier(n_estimators = FOREST_TREES_COUNT)
 rf.fit(X_input, y_target)

 #-----------------------------------------------------------------------
 # testing
 load_csv_train = loadData('test.csv')  # create an instance of loadData
 # testing_data = load_csv_train.loadCSVFull()
 testing_data = load_csv_train.loadCSVFewRows(NUMBER_ROWS)
 testArr = testing_data.as_matrix() 
 predicted_probs = rf.predict_proba(testArr)
 print(predicted_probs)
 # predicted_probs = ["%f" % x[1] for x in predicted_probs]
 # csv_io.write_delimited_file("random_forest_solution.csv", predicted_probs) 

  



