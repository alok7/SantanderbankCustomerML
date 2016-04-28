'''
var38 var15, are important features.
So we would lay much emphasis on analysis of these two features
'''



#import pandas a data processing and CSV file I/O library
import pandas as pd
import numpy as np
# Seaborn -- a python graphing library
import seaborn as sns 
import matplotlib.pyplot as plt 
#we perform a chi^2 test to retrieve the two best features
# from sklearn.feature_selection import SelectBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif

#------------------------------------------------------------------#
#------------------------------------------------------------------#
#---------------------Class for reading CSV------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#

class loadData:
 """Common base class for the Santander csv file reader"""
 def __init__(self, path):
  self.path = path
 
 def loadCSVFull(self):
  customer = pd.read_csv(self.path)
  return customer
 
 def loadCSVSkipHeader(self):
  '''skiprows = int, means number of rows to skip from the start of the file.
     skiprows = list of ints, means the these index rows to skip'''
  customer = pd.read_csv(self.path, skiprows = 1)
  return customer

 def loadCSVSkipFewRows(self, number_skiprows, number_readrows):
  customer = pd.read_csv(self.path, skiprows = number_skiprows,nrows = number_readrows)
  return customer
 
 def loadCSVFewRows(self, number_readrows):
  customer = pd.read_csv(self.path, nrows = number_readrows)
  return customer
 
 def loadCSVFewCols(self, to, frm):
  ''' to is 0 index'''
  customer = pd.read_csv(self.path, usecols = range(to, frm+1))
  return customer

#------------------------------------------------------------------#
#------------------------------------------------------------------#
#-----Class for getting various statistics for feature vector------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#

class dataStatistics:
 """common class for attributes' various statistics"""
 
 def __init__(self, csvReader):
  self.train = csvReader
 
 def describeUnhappyCustomer(self, attr):
  self.train.loc[self.train['TARGET']==1, attr].describe()
 
 def describeHappyCustomer(self, attr):
  self.train.loc[self.train['TARGET']==0, attr].describe()
 
 def getAttributeMean(self, attr):
  return self.train[attr].mean()
 
 def describeAttribute(self, attr):
  return self.train[attr].describe()
 
 def getAttributeMin(self, attr):
  return self.train[attr].min()
 
 def getAttributeMax(self, attr):
  return self.train[attr].max()
 
 def getAttributeVariance(self, attr):
  return self.train[attr].var()
 
 '''Square root of variance''' 
 def getAttributeDeviaion(self, attr):
  return self.train[attr].std()
 
 def getAttributeMedian(self, attr):
  return self.train[attr].median()
 
 '''Calculate pairwise correlation of two column vectors'''
 def correlationOfTwoAtrributes(self, attr1='var38', attr2='var15'):
  return self.train[[attr1, attr2]].corr()
 
 def correlationMatrix(self):
  return self.train.corr()
 
 def covarianceMatrxi(self):
  return self.train.cov()
 
 def getAttributeSum(self, attr): 
  return self.train[attr].sum()
 
 def getAttributeSkew(self, attr): 
  return self.train[attr].skew()
 
 '''Most Common value count for feature vector attr'''
 def getAttributeValueCount(self, attr):
  return self.train[attr].value_counts()

#------------------------------------------------------------------#
#------------------------------------------------------------------#
#---------Class for Graphical Analysis of training data------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#

class createMap:
 """Common class for plotting data"""
 def __init__(self, csvReader):
  self.train = csvReader
 
 '''
 A histogram is a graphical representation of the distribution of numerical data.
 It gives a rough estimate of the probability distribution
 of the underlying distribution of the data 
 '''
 '''Similar below analysis for var15'''
 
 def Hist(self, attr='var38'):
  self.train[attr].hist(bins=1000)
  plt.title(' Histogram of ' + str(attr))
  plt.savefig(' Histogram of ' + str(attr))
  plt.close()
  # plt.show()
 
 def LogHist(self, attr='var38'):
  self.train[attr].map(np.log).hist(bins=700)
  plt.title('Log Histogram of ' + str(attr))
  plt.savefig('Log Histogram of ' + str(attr))
  plt.close()
  # plt.show()
 
 '''Find most common value usingvalue_counts()'''
 def LogHistExcludingCommonValue(self, attr='var38'):
  self.train.loc[~np.isclose(self.train[attr], 117310.979016), attr].map(np.log).hist(bins=100)
  plt.title(' ')
  # plt.show()
 
 # Let's look at the density of the age of happy/unhappy customers
 def densityPlot(self, attr='var15'):
  sns.FacetGrid(self.train, hue="TARGET", size=6).map(sns.kdeplot, attr).add_legend()
  plt.title('Unhappy customers are older than happy ones')
  plt.savefig('Unhappy customers are older than happy ones')
  plt.close()
  # plt.show()

 def scatterPlot(self, attr1= 'var38', attr2='var15'):
  sns.FacetGrid(self.train, hue="TARGET", size=10).map(plt.scatter, attr1, attr2).add_legend()
  plt.title('Scatter plot between ' + str(attr1) + str(attr2))
  plt.savefig('Scatter plot between ' + str(attr1) + str(attr2))
  plt.close()
  # plt.show()

 def scatterLogPlot(self, attr1= 'var38', attr2='var15'):
  sns.FacetGrid(self.train, hue="TARGET", size=10).map(plt.scatter, "logvar38", "var15").add_legend()
  plt.ylim([0,120])
  plt.title('Scatter Log plot between ' + str(attr1) + str(attr2))
  plt.savefig('Scatter Log plot between ' + str(attr1) + str(attr2))
  plt.close()
  # plt.show()
 
 def pairPlot(self, lst = ['var15','var36','TARGET']):
  # Each variable in lst is along y-axis
  sns.pairplot(self.train[lst], hue="TARGET", size=2, diag_kind="kde")
  plt.title(' ')
  plt.savefig('Pairwise plot')
  plt.close()
  # plt.show()
 def heatMap(self, data):
  sns.heatmap(data)
  plt.title(' HeatMap of covariance Matrix ')
  plt.savefig('Covariance Matrix')
  plt.close()
  # plt.show()
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------Class for feature selection---------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#

class featureSelection:
  """Common class for feture selections"""
  def __init__(self, csvReader):
   self.training = csvReader

  def getFeatures(self, number_of_features=10):
   # X = self.training.iloc[:,:,-1]
   y = self.training['TARGET']
   X = self.training.drop(['TARGET'], axis=1)
   #Select features according to the k highest scores.
   #selectFeatures = SelectBest(chi2, k= number_of_features)
   #Select the best 10 percentile
   # We can use other classifier as well for Selection like chi2
   selectFeatures = SelectPercentile(f_classif, percentile= number_of_features)
   selectFeatures.fit(X, y)
   # X_select = selectFeatures.transform(X)
   features = selectFeatures.get_support(indices=True)
   # print("Best feature: "+ features[0])
   return(features) 


#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#
#------------------------------------------------------------------#

# if __name__== '__main__':
def executethisModule(): 
 #-------------------------------------------------------------
 data = loadData('train.csv')
 # partial_data = data.loadCSVFewRows(10000)
 full_data = data.loadCSVFull()
 print("Module for all the Santander Data visualization")
 #---------------------------------------------------------------
 # features_ = featureSelection(partial_data)
 # best_features  = features_.getFeatures(10)
 # print(best_features)
 #--------------------------------------------------------------
 # stat = dataStatistics(full_data)
 # print(stat.describeAttribute('var38'))
 # print(stat.getAttributeMean('var38'))
 # This gives the correlation between each column and each other column
 # df = stat.correlationOfTwoAtrributes('var38', 'var15')
 # Since the result of this is a dataframe, we can index it and only get the correlations for the var38 column  
 # print(df['var38'])
 # plt.scatter(df['var38'], df['var15'])
 # plt.show()
 #---------------------------------------------------------------
 graphics_ = createMap(full_data)
 graphics_.Hist('var38')
 graphics_.LogHist('var38')
 graphics_.densityPlot('var15') 
 graphics_.scatterPlot('var38', 'var15')
 graphics_.pairPlot(['var15','var36','TARGET'])
 # graphics_.heatMap(full_data)