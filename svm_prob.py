"This script is written to write in everything in terms of numpy arrays. Earlier I used to used lists but that is not convenient."
"In this script probabilistic outputs are enabled in the sense that it gives the distance from the hyperplane. probability=True is not done because it does five fold cross validation which is extremely expensive."

import csv
from math import floor
import numpy as np
from sklearn import svm
from sklearn import cross_validation
#import matplotlib.pyplot as plt
import time
from auxiliary_functions import *

"This function takes the input from the file and saves an output file output.csv. We will take one by one line of testing and save the output for that."
start_time = time.time()

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

# testing input
with open('test.csv', 'r') as csvfile:
    customer_read = csv.reader(csvfile)
    for (row_num,row) in enumerate(customer_read):
        if row_num == 0:
            field_names = row
        else:
            customer_test.append(row_convert_float(row))
print('test.csv has been read')    

# Removing the ID column from test data
for customer in customer_test:
    del customer[0]    

customer_test_arr = np.array(customer_test)
normalize_stats(customer_test_arr,stats)
customer_test = None

clf = svm.SVC(C=1,kernel='rbf',verbose=True,class_weight='balanced',gamma=0.1,probability=False)
clf.fit(customer_train_input, customer_train_output)
customer_output = clf.decision_function(customer_test_arr)
np.save('svm_object.npy',clf)

print("--- %s seconds ---" % (time.time() - start_time))
np.savetxt('test_predicted_prob.txt',customer_output)
print("--- %s seconds ---" % (time.time() - start_time))
