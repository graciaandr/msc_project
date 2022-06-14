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
df_beta_values = pd.read_csv('./data/classifying_data/CLL_study_filt-beta-values.txt', sep = ';')
# df_beta_values = pd.read_csv('./classifying_data/artistic_study_filt-beta-values.txt', sep = ';')

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
df = df.apply(pd.to_numeric)

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.drop(['label'], axis=1)
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train1, X_valid, y_train1, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)


# Hyper Parameter Tuning- finding the best parameters and kernel
# Performance tuning using GridScore
# print('Hyper Parameter Tuning')
# param_grid = {'C': [int(x) for x in np.linspace(start = 1, stop = 1000, num = 100)], 
#               'gamma': list(np.arange (start = 0.01, stop = 0.1, step = 0.01)),
#               'kernel': ['linear', 'rbf', 'sigmoid'],
#               'degree': [int(x) for x in np.linspace(start = 1, stop = 10, num = 1)]
#    }

# svr = svm.SVC()
# clf = GridSearchCV(svr, param_grid,cv=5)
# clf.fit(X_train, y_train)

# print('the best params are:')
# print(clf.best_params_)

# add here as comment the best params 
# {'C': 1, 'degree': 1, 'gamma': 0.01, 'kernel': 'linear'}

# using the optimal parameters, initialize and train SVM classifier
clf = svm.SVC(kernel= 'linear', degree = 1, gamma = 0.01, C = 1, class_weight='balanced', probability=True, random_state=42)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('./scratch/ROC_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/ROC_SVM_all_features.png')
plt.close()
# plt.show()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'CIN2+']); ax.yaxis.set_ticklabels(['Control', 'CIN2+']);
plt.savefig('./scratch/cf_matrix_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_SVM_all_features.png')
plt.close()
# plt.show()

# cf matrix with percentages
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'CIN2+']); ax.yaxis.set_ticklabels(['Control', 'CIN2+']);
plt.savefig('./scratch/cf_matrix_perc_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_perc_SVM_all_features.png')
plt.close()
# plt.show()