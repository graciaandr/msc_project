import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


# load data sets
df_beta_values = pd.read_csv('../data/classifying_data/CLL_study_filt-beta-values.txt', sep = ';')
# df_beta_values = pd.read_csv('./classifying_data/CLL_study_filt-beta-values.txt', sep = ';')

# transpose data matrix 
df_beta_transposed = df_beta_values.transpose() 
df_beta_transposed.index.name = 'old_column_name' ## this is to make filtering easier later
df_beta_transposed.reset_index(inplace=True)

# try imputing with several imputation methods
# impute ctrls with ctrls and cases with cases
imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value=50)
 
# extract and add column with labels (0,1) for control and treated samples
df_ctrl = df_beta_transposed.loc[lambda x: x['old_column_name'].str.contains(r'(ctrl)')]
df_ctrl = df_ctrl.drop(['old_column_name'], axis=1)
imputer1 = imputer.fit(df_ctrl)
imputed_df_ctrl = imputer1.transform(df_ctrl)
df_ctrl_new = pd.DataFrame(imputed_df_ctrl, columns = df_ctrl.columns)
df_ctrl_new.loc[:, 'label'] = 0

df_trt = df_beta_transposed.loc[lambda x: x['old_column_name'].str.contains(r'(case)')]
df_trt = df_trt.drop(['old_column_name'], axis=1)
imputer2 = imputer.fit(df_trt)
imputed_df_trt = imputer2.transform(df_trt)
df_trt_new = pd.DataFrame(imputed_df_trt, columns = df_trt.columns)
df_trt_new.loc[:, 'label'] = 1

# merge trt and ctrl data frames
df = pd.concat([df_trt_new, df_ctrl_new])

# Resampling the minority class. The strategy can be changed as required. (source: https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/)
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df.drop('label', axis=1), df['label'])
df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.drop(['label'], axis=1)
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


# Hyper Parameter Tuning- finding the best parameters and kernel
# Performance tuning using GridScore

param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svr = svm.SVC()
clf = GridSearchCV(svr, param_grid,cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)

# using the optimal parameters, initialize and train SVM classifier
  # clf = svm.SVC(kernel= 'rbf', gamma = 0.001, C = 100)
clf = svm.SVC(kernel= 'linear', gamma = 0.001, C = 1, class_weight='balanced', probability=True)
# clf = svm.SVC(kernel= 'linear', gamma = 0.001, C = 1)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# return accuracy and precision score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('../scratch/ROC_SVM_all_features.png')
plt.close()
# plt.show()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity : ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity (should be same as recall score): ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

sns.heatmap(cf_matrix, annot=True, fmt='.3g')
plt.savefig('../scratch/cf_matrix_SVM_all_features.png')
plt.close()
# plt.show()

# cf matrix with percentages
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.savefig('../scratch/cf_matrix_perc_SVM_all_features.png')
plt.close()
# plt.show()