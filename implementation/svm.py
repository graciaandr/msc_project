from cmath import nan
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
# from sklearn.preprocessing import SimpleImputer
from sklearn.impute import SimpleImputer


# load data sets
# took example rrbs data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104998
# df_m_values = pd.read_csv('data/classifying_data/filt-m-values.txt', sep = ';')
df_beta_values = pd.read_csv('data/classifying_data/filt-beta-values.txt', sep = ';')

print(df_beta_values.head(5))
print( 'path: ',os.getcwd())

# transpose and add column with labels (0,1) for control and treated samples
df_beta_transposed = df_beta_values.transpose() 

print(df_beta_transposed.head(3))
df_ctrl = df_beta_transposed.loc[["ctrl1", "ctrl3", "ctrl4", "ctrl5"]]
df_ctrl['label'] = 0

df_trt = df_beta_transposed.loc[["case1", "case2", "case3", "case5"]]
df_trt['label'] = 1

# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])
print(df.head(5))
print(df.shape)

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.loc[:, df.columns != 'label']
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('split data into training and testing data')

# check for NAs 
print(df.isnull().sum())

## use SimpleImputer from scikit learn to impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)

# initialize and train SVM classifier
clf = svm.SVC()
clf = clf.fit(X_train_imp, y_train)

# apply SVM to test data
y_pred = clf.predict(X_test_imp)

print('classifier has been trained')
stopppp

# return accuracy and precision score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.3g')
plt.show()

# cf matrix with percentages
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.show()

