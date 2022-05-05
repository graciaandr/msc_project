from cmath import nan
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.impute import SimpleImputer

# load data sets
# df_m_values = pd.read_csv('data/classifying_data/filt-m-values.txt', sep = ';')
df_beta_values = pd.read_csv('data/classifying_data/filt-beta-values.txt', sep = ';')
print( 'path: ',os.getcwd())

# transpose data matrix 
df_beta_transposed = df_beta_values.transpose() 
df_beta_transposed.index.name = 'old_column_name' ## this is to make filtering easier later
df_beta_transposed.reset_index(inplace=True)
df_beta_transposed = df_beta_transposed.replace(np.nan, 0.5) # replace NA with 'neutral' value ?

# still what to do with NAs???
df_beta_transposed.dropna(axis='columns',  inplace=True)
print(df_beta_transposed.head(5))

# extract and add column with labels (0,1) for control and treated samples
df_ctrl = df_beta_transposed.loc[lambda x: x['old_column_name'].str.contains(r'(ctrl)')]
df_ctrl['label'] = 0

df_trt = df_beta_transposed.loc[lambda x: x['old_column_name'].str.contains(r'(case)')]
df_trt['label'] = 1

# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])
print(df)
print(df.shape)

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.drop(['old_column_name','label'], axis=1)
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
print('split data into training and testing data')

# check for NAs:
print(df.isnull().sum())

# initialize and train SVM classifier
clf = RandomForestClassifier(max_depth=20, random_state=0)
clf = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = clf.predict(X_test)
print('Status: classifier has been trained')

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

