import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os


# load data sets
# took example rrbs data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104998
df_m_values = pd.read_csv('data/classifying_data/m-values.txt', sep = ';')
df_beta_values = pd.read_csv('data/classifying_data/beta-values.txt', sep = ';')

print(df_m_values.head(5))
print( 'path: ',os.getcwd())

# add column with labels (0,1) for control and treated samples
df_ctrl = df_m_values[['case1', 'case2', 'case6']]
df_ctrl['label'] = 0

df_trt = df_m_values[['ctrl2', 'ctrl2']]
df_trt['label'] = 1

# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])
print(df.shape)
# print(df)

stopppp

# keep numeric values - for classification 
df_num = df[['coverage', 'freqC', 'freqT', 'label']]

X = df[['coverage', 'freqC', 'freqT']]
y = df_num['label']

print('here')

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('split data into training and testing data')

# initialize and train SVM classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
print('classifier has been trained')

# apply SVM to test data
y_pred = clf.predict(X_test)

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

