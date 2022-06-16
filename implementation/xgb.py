import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# load data sets
df_beta_values = pd.read_csv('./data/classifying_data/CLL_study_filt-beta-values.txt', sep = ';')
# df_beta_values = pd.read_csv('./classifying_data/CLL_study_filt-beta-values.txt', sep = ';')

# transpose data matrix 
df_beta_transposed = df_beta_values.transpose() 
df_beta_transposed.index.name = 'old_column_name' ## this is to make filtering easier later
df_beta_transposed.reset_index(inplace=True)

# try imputing with several imputation methods
# impute ctrls with ctrls and cases with cases
imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value= 50)
 
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
df = df.apply(pd.to_numeric)
# df.to_csv('beta_vals_labelled_data.txt', index=False, index_label=None, sep = ";", header=True)

# Resampling the minority class. The strategy can be changed as required. (source: https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/)
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df.drop('label', axis=1), df['label'])
df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.drop(['label'], axis=1)
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# ## XGB Hyperparameter Tuning
# parameters = {'max_depth': [int(x) for x in np.linspace(start = 10, stop = 100, num = 50)],
#               # 'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 100, num = 50)],
#               'learning_rate': list(np.arange (start = 0.001, stop = 0.1, step = 0.001)),
#               'sampling_method': ['uniform', 'gradient_based'],
#               }

# DTC = DecisionTreeClassifier(random_state = 42, max_features = "auto", class_weight = "balanced", max_depth = None)
# XGB = xgb.XGBClassifier(base_estimator = DTC)

# # run grid search
# xgb_random = GridSearchCV(estimator = XGB, param_grid=parameters, scoring = 'roc_auc', refit=False)
# xgb_random.fit(X_train, y_train)

# print(xgb_random.best_params_)

# Output
# {'learning_rate': 0.08, 'max_depth': 10, 'n_estimators': 100, 'sampling_method': 'uniform'}
 

## train XGB classifier
clf = xgb.XGBClassifier(learning_rate = 0.08, max_depth = 10, sampling_method = 'uniform', n_estimators=100, random_state=42) # need to adjust according to what the best parameters are
fit = clf.fit(X_train, y_train)

# apply XGB to test data
clf = xgb.XGBRFClassifier(n_estimators=100, learning_rate=0.001, max_depth = 100)

fit = clf.fit(X_train,y_train)

y_pred = fit.predict(X_test)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('./scratch/ROC_xgb_all_features.png')
# plt.savefig('./figures/ROC_xgb_all_features.png')
plt.close()
plt.show()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix__xgb_all_features.png')
# plt.savefig('./figures/cf_matrix__xgb_all_features.png')
plt.close()
plt.show()

# cf matrix with percentages
ax= plt.subplot()
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix_perc_xgb_all_features.png')
# plt.savefig('./figures/cf_matrix_perc_xgb_all_features.png')
plt.close()
plt.show()

# Feature Selection 
features = list(df.columns)
f_i = list(zip(features,clf.feature_importances_))
f_i.sort(key = lambda x : x[1])
f_i = f_i[-50:]
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.savefig('./scratch/feature_selection_XGB.png')
# plt.savefig('./figures/feature_selection_XGB.png')
plt.close()
plt.show()

first_tuple_elements = []
for a_tuple in f_i:
	first_tuple_elements.append(a_tuple[0])
first_tuple_elements.append('label')

# subset of data frame that only includes the n selected features
df_selected = df[first_tuple_elements]

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df_selected.drop(['label'], axis=1)
y = df_selected.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize and train SVM classifier
clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('./scratch/ROC_xgb_sel_features.png')
# plt.savefig('./figures/ROC_xgb_sel_features.png')
plt.close()
# plt.show()

## calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
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
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix_xgb_sel_features.png')
# plt.savefig('./figures/cf_matrix_xgb_sel_features.png')
plt.close()
# plt.show()

# cf matrix with percentages
ax= plt.subplot()
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix_perc_xgb_sel_features.png')
# plt.savefig('./figures/cf_matrix_perc_xgb_sel_features.png')
plt.close()
# plt.show()