"In this script randomly chosen 10000 samples will be taken for cross validation instead of the first 10000. Also output will be probabilities instead of classes."
import csv
from math import floor
import numpy as np
from sklearn import svm, grid_search
from sklearn import cross_validation
#import matplotlib.pyplot as plt
import time
from auxiliary_functions import *
from random import sample

customer_train = []
customer_test = []

#training input
with open('train.csv', 'r') as csvfile:
    customer_read = csv.reader(csvfile)
    for (row_num,row) in enumerate(customer_read):
        if row_num == 0:
            field_names = row
        else:
            customer_train.append(row_convert_float(row))
print('Train.csv has been read')

# Removing the ID column from train data
for customer in customer_train:
    del customer[0]        

customer_train_arr = np.array(customer_train)
customer_train_input = customer_train_arr[:,:-1]
customer_train_output = customer_train_arr[:,-1]
stats = normalize(customer_train_input)
customer_train = None
customer_train_arr = None
no_samples = np.shape(customer_train_output)[0]#expected to be about 70,000
no_cross_validate = 10000
ind_cross_validate = sample(range(no_samples),no_cross_validate)

c_range = [1,0.1,0.001,0.0001];
gam_range = [0.0001,0.001,0.01,0.1];
cross_scores = np.zeros((len(c_range),len(gam_range)))
parameters = [{'kernel':['rbf'], 'C':c_range, 'gamma':gam_range,'verbose':[True]}]

np.save('c_arr_1.npy',c_range)
np.save('gam_arr_1.npy',gam_range)
np.savetxt('c_arr_1.txt', c_range)
np.savetxt('gam_arr_1.txt', gam_range)

svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(customer_train_input[ind_cross_validate],customer_train_output[ind_cross_validate])
np.savetxt('best_params.txt',clf.best_params_)
