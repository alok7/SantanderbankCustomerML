import csv
import numpy as np
import math
from auxiliary_functions import sigmoid_array

predict = []
predict_int = []
customer_test = []
customer_ids = []

with open('test_predicted_prob.txt', 'r') as csvfile:
    predict_read = csv.reader(csvfile)
    for row in predict_read:
        predict.append(row[0])

predict_dist = np.array(predict).astype(np.float)
predict_dist_centered = predict_dist - np.average(predict_dist)
predict_dist_norm = predict_dist_centered/(2*max(abs(predict_dist_centered))) + 0.5
#predict_probs = sigmoid_array(predict_dist_centered)
# +0.5 was for shifting it from 0 to 1

# testing input
with open('test.csv', 'r') as csvfile:
    customer_read = csv.reader(csvfile)
    for (row_num,row) in enumerate(customer_read):
        if row_num == 0:
            field_names = row
        else:
            customer_test.append(row)

for customer in customer_test:
    customer_ids.append(int(customer[0]))


np.savetxt("output_prob.csv", zip(np.asarray(customer_ids),predict_dist_norm), delimiter=',',fmt=['%u','%.2f'])
