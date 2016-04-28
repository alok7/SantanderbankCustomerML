import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

def executethisModule():
 #Loading training data
 start_time=time.time();
 x_train=np.asarray(np.loadtxt(open("train.csv","rt"),delimiter=",",skiprows=1));
 n=len(x_train);
 p=len(x_train[0])-1;

 #Preprocessing training data
 y_train=np.reshape(x_train[:,p],(n,1)); #Extracting labels
 x_train=x_train[:,1:p-1] #Deleting id and labels columns from training data
 p=p-1; #number of features used for classification
 x0=x_train[:,0];
 x0[x0==-999999]==2; #Replacing -999999 by 2
 x_train[:,0]=x0
 scaler = StandardScaler().fit(x_train)
 x=scaler.transform(x_train) #Normalising training data
 del x_train, x0;

 clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=60,max_features=0.2,class_weight='balanced');
 #Uncomment the following portion for cross validation
 ##parameters = {'criterion':['gini'], 'max_depth':[40,50,60,70], 'max_features':[0.2,0.4,0.6,0.8,1.0], 'class_weight':['balanced']}
 ##est = tree.DecisionTreeClassifier();
 ##clf = grid_search.GridSearchCV(est, parameters, cv=5)
 clf.fit(x,np.ravel(y_train));
 y_pred=np.reshape(clf.predict(x),(n,1));
 y_prob=clf.predict_proba(x);
 dist=np.abs(y_prob[0,:]-y_pred[0]);
 ind=np.argmin(dist);

 #err=np.sum(np.abs(y_train-y_pred))/n;
 met=roc_auc_score(y_train,y_prob[:,ind]);
 print("Area under the ROC curve for training data: ");
 print(met);

 del x;

 #Plotting feature importances
 w=clf.feature_importances_
 plt.plot(w)
 plt.ylabel('Feature Importances')
 plt.xlabel('Feature Index')
 plt.show()

 #Loading test data
 x_test=np.asarray(np.loadtxt(open("test.csv","rt"),delimiter=",",skiprows=1));
 n_test=len(x_test);

 #Preprocessing test data
 id_test=x_test[:,0];
 x_test=x_test[:,1:p]; #Deleting id column from test data
 x0=x_test[:,0];
 x0[x0==-999999]==2; #Replacing -999999 by 2
 x_test[:,0]=x0;
 x_test=scaler.transform(x_test) #Normalising test data
 y_test=clf.predict(x_test);
 with open('output_dtree.csv', 'wb') as csvfile:
  writer=csv.writer(csvfile,dialect='excel',delimiter=',')
  writer.writerow(["ID"]+["TARGET"])
  for i in range(n_test):
   writer.writerow([int(id_test[i])]+[int(y_test[i])])

 print("Time to run in seconds: ");
 print(time.time()-start_time);
