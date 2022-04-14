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
df1 = pd.read_csv('../data/YB5_CRC_study/GSM2813719_YB5_DMSO4d1.txt', sep = '\t')
df2 = pd.read_csv('../data/YB5_CRC_study/GSM2813720_YB5_DMSO4d2.txt', sep = '\t')
df3 = pd.read_csv('../data/YB5_CRC_study/GSM2813721_YB5_DMSO4d3.txt', sep = '\t')

df4 = pd.read_csv('../data/YB5_CRC_study/GSM2813722_YB5_HH1_10uM4d1.txt', sep = '\t')
df5 = pd.read_csv('../data/YB5_CRC_study/GSM2813723_YB5_HH1_10uM4d2.txt', sep = '\t')
df6 = pd.read_csv('../data/YB5_CRC_study/GSM2813724_YB5_HH1_10uM4d3.txt', sep = '\t')

df7 = pd.read_csv('../data/YB5_CRC_study/GSM2813725_YB5_con1.txt', sep = '\t')
df8 = pd.read_csv('../data/YB5_CRC_study/GSM2813726_YB5_con2.txt', sep = '\t')

df_diff_methyl_CPG = pd.read_csv('data/YB5_CRC_study/filtered_CPGs.txt', sep = '\t')
print(df_diff_methyl_CPG.head(5))
# print( 'path: ',os.getcwd())

# add column with labels (0,1) for control and treated samples
# df_trt = pd.concat([df1, df2, df3, df4, df5, df6])
df_trt = (df1.head(50000)).copy()

df_trt_by_dms = df_trt.loc[(df_diff_methyl_CPG.start.isin(df_trt.base)) & (df_diff_methyl_CPG.chr.isin(df_trt.chr)),:].drop_duplicates()
# df.loc[(df.ID1.isin(df1.ID1))&(df.ID2.isin(df1.ID2)),:].drop_duplicates()

df_trt['label'] = 1
print(df_trt_by_dms)
print(df_trt_by_dms.shape)

stopppp
# df_ctrl = pd.concat([df7, df8])
df_ctrl = (df7.head(50000)).copy()
df_ctrl['label'] = 0

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

