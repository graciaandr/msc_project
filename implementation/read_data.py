import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


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

# add column with labels (0,1) for control and treated samples
# df_trt = pd.concat([df1, df2, df3, df4, df5, df6])
df_trt = df1 # .head(1000)
df_trt['label'] = 1

# df_ctrl = pd.concat([df7, df8])
df_ctrl = df7 # .head(1000)
df_ctrl['label'] = 0

# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])

print(df.head(5))
print(df.shape)
# print(set(df['chr']))
# print(set(df['strand']))

# keep numeric values - for classification 
df_num = df[['coverage', 'freqC', 'freqT', 'label']]
print(df_num.head(5))
print(df_num.tail(5))

X = df[['coverage', 'freqC', 'freqT']]
y = df_num['label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize
clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))

# Plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True)
plt.show()

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.show()

