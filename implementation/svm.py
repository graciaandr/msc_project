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
import pickle

# load training data set
df_train = pd.read_csv('./data/classifying_data/training_data_ARTISTIC_trial.csv', sep = ";")
df_y_train = pd.read_csv('./data/classifying_data/labels_training_data_ARTISTIC_trial.csv', sep = ";")
# df_train = pd.read_csv('./classifying_data/training_data_ARTISTIC_trial.csv', sep = ";")
# df_y_train = pd.read_csv('./classifying_data/labels_training_data_ARTISTIC_trial.csv', sep = ";")

# load testing data set
df_test = pd.read_csv('./data/classifying_data/testing_data_ARTISTIC_trial.csv', sep = ";")
df_y_test = pd.read_csv('./data/classifying_data/labels_testing_data_ARTISTIC_trial.csv', sep = ";")
# df_test = pd.read_csv('./classifying_data/testing_data_ARTISTIC_trial.csv', sep = ";")
# df_y_test = pd.read_csv('./classifying_data/labels_testing_data_ARTISTIC_trial.csv', sep = ";")

# load validation data set
df_val = pd.read_csv('./data/classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
df_y_val = pd.read_csv('./data/classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")
# df_val = pd.read_csv('./classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
# df_y_val = pd.read_csv('./classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")

X_train = np.array(df_train)
X_test = np.array(df_test)
X_val = np.array(df_val)

y_train = np.array(df_y_train)
y_test = np.array(df_y_test)
y_val = np.array(df_y_val)

## Hyper Parameter Tuning- finding the best parameters and kernel
## Performance tuning using GridScore
# print('Hyper Parameter Tuning')
# param_grid = {'C': [int(x) for x in np.linspace(start = 1, stop = 100, num = 10)], 
#               'gamma': list(np.arange (start = 0.01, stop = 0.1, step = 0.01)),
#               'kernel': ['linear', 'rbf', 'sigmoid'],
#               'degree': [int(x) for x in np.linspace(start = 1, stop = 10, num = 1)]
#    }

# svr = svm.SVC()
# clf = GridSearchCV(svr, param_grid,cv=5, scoring='recall')
# clf.fit(X_train, y_train)

# print('the best params are:')
# print(clf.best_params_)

# # the best params are:
# # {'C': 1, 'degree': 1, 'gamma': 0.01, 'kernel': 'linear'}


# using the optimal parameters, initialize and train SVM classifier
clf = svm.SVC(kernel= 'linear', degree = 1, gamma = 0.01, C = 1, 
              class_weight='balanced', probability=True, random_state=20)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# save the model to disk
filename = 'svm_model.sav'
pickle.dump(fit, open(filename, 'wb')) 

print("########## TEST DATA SET ##########")
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
# plt.savefig('./scratch/ROC_SVM_all_features.png')
# plt.savefig('./figures/ROC_SVM_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))


plt.figure(figsize=(6.5, 5))
ax = plt.subplot()
# plt.legend(fontsize='14')
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r', annot_kws={"size":16})
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font); ax.set_ylabel('True labels', fontdict=label_font); 
# ax.set_title('Confusion Matrix'); 
ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_SVM_all_features.png')
# plt.savefig('./figures/cf_matrix_SVM_all_features.png')
# plt.show()
plt.close()

# # cf matrix with percentages
# plt.figure(figsize=(6.5, 5))
# ax= plt.subplot()
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues')
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# # ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# # plt.savefig('./scratch/cf_matrix_perc_SVM_all_features.png')
# plt.savefig('./figures/cf_matrix_perc_SVM_all_features.png')
# plt.close()
# # plt.show()


### feature importances
df = pd.read_csv('./data/classifying_data/complete_data_ARTISTIC_trial.csv', sep = ";")
# df = pd.read_csv('./classifying_data/complete_data_ARTISTIC_trial.csv', sep = ";")
features = list(df.columns [df.columns != "label"])

top_n_features = 75 
coef = clf.coef_.ravel()
# print("top features in abs numbers",np.argsort(abs(coef))[-top_n_features:])
# top_positive_coefficients = np.argsort(coef)[-top_features:]
# top_negative_coefficients = np.argsort(coef)[:top_features]
# top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
top_abs_coefs = np.argsort(abs(coef))[-top_n_features:]
# top_abs_coefs = np.argsort(abs(coef))
# top_abs_coefs = top_abs_coefs[-top_n_features:]

feat_list = [features[i] for i in top_abs_coefs]
df_features = pd.DataFrame(feat_list, columns = ["SVM_features"])
df_features ["importance"] = coef[top_abs_coefs]
print(df_features)
# df_features.to_csv('./data/classifying_data/SVM_features.csv', sep = ";", header=True)
# df_features.to_csv('./classifying_data/SVM_features.csv', sep = ";", header=True)


# print(coef[top_abs_coefs])
# print(feat_list)

result = sum(abs(number) for number in coef[top_abs_coefs])
print('Sum of feature importance', (result))

def plot_coefficients(classifier, feature_names, top_n_features=75):
     coef = classifier.coef_.ravel()
     top_abs_coefs = np.argsort(abs(coef))[-top_n_features:]

     
     # create plot
     plt.figure(figsize=(15, 15))
     colors = ['red' if c < 0 else 'blue' for c in coef[top_abs_coefs]]
     plt.bar(np.arange(top_n_features), coef[top_abs_coefs], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(0, top_n_features), feature_names[top_abs_coefs], rotation=30, ha='right')
     # plt.savefig('./scratch/transposed_feature_selection_SVM.png')
     # plt.savefig('./figures/transposed_feature_selection_SVM.png')
     # plt.show()
     plt.close()
     
plot_coefficients(clf, features, top_n_features)

# subset of data frame that only includes the n selected features
feat_list.append('label')
df_selected = df[feat_list]

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df_selected.drop(['label'], axis=1)
y = df_selected.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

# initialize and train SVM classifier
# clf = GradientBoostingClassifier(n_estimators = 92, max_depth = 15, learning_rate = 0.06999999999999999, max_features = 'log2', min_samples_split = 2, random_state=20)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# save the model to disk
filename = 'FS_svm_model.sav'
pickle.dump(fit, open(filename, 'wb')) 


print("########## TEST DATA SET - FEATURE SELECTION ##########")

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
# plt.savefig('./scratch/ROC_SVM_sel_features.png')
# plt.savefig('./figures/ROC_SVM_sel_features.png')
# plt.show()
plt.close()

## calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))


plt.figure(figsize=(6.5, 5))
ax = plt.subplot()
# plt.legend(fontsize='14')
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r', annot_kws={"size":16})
label_font = {'size':'15'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font); ax.set_ylabel('True labels', fontdict=label_font); 
# ax.set_title('Confusion Matrix'); 
ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust to fit
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_SVM_sel_features.png')
# plt.savefig('./figures/cf_matrix_SVM_sel_features.png')
# plt.show()
plt.close()

# # cf matrix with percentages
# plt.figure(figsize=(6.5, 5))
# ax= plt.subplot()
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues')
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# # ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# # plt.savefig('./scratch/cf_matrix_perc_SVM_sel_features.png')
# plt.savefig('./figures/cf_matrix_perc_SVM_sel_features.png')
# plt.close()
# # plt.show()