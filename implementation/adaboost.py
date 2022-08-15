import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
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

### Machine Learning 

## Adaboost Hyperparameter Tuning
# parameters = {'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 100, num = 10)],
#               'learning_rate': list(np.arange (start = 0.01, stop = 0.1, step = 0.01)),
#               }

# DTC = DecisionTreeClassifier()
# ABC = AdaBoostClassifier(random_state = 20, base_estimator=DTC)

# # ## run grid search
# ada_random = GridSearchCV(estimator = ABC, param_grid=parameters, scoring = 'recall', refit=False)
# ada_random.fit(X_train, y_train)

# print(ada_random.best_params_)
# print('\n')

# stop0

### Output: 
### {'learning_rate': 0.001, 'n_estimators': 50} ????

# train adaboost classifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate = 0.09, random_state=20)
fit = clf.fit(X_train, y_train)

# apply adaboost classifier to test data
y_pred = fit.predict(X_test)

# save the model to disk
filename = 'adaboost_model.sav'
pickle.dump(fit, open(filename, 'wb')) 

print("########## TEST DATA SET ##########")

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
# plt.savefig('./scratch/ROC_adaboost_all_features.png')
# plt.savefig('./figures/ROC_adaboost_all_features.png')
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
# plt.savefig('./scratch/cf_matrix__adaboost_all_features.png')
# plt.savefig('./figures/cf_matrix__adaboost_all_features.png')
# plt.show()
plt.close()

# # cf matrix with percentages
# ax= plt.subplot()
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues')
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# # ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# # plt.savefig('./scratch/cf_matrix_perc_adaboost_all_features.png')
# plt.savefig('./figures/cf_matrix_perc_adaboost_all_features.png')
# plt.close()
# # plt.show()

# Feature Selection 
df = pd.read_csv('./data/classifying_data/complete_data_ARTISTIC_trial.csv', sep = ";")
# df = pd.read_csv('./classifying_data/complete_data_ARTISTIC_trial.csv', sep = ";")
features = list(df.columns)
f_i = list(zip(features,clf.feature_importances_))
f_i.sort(key = lambda x : x[1])
f_i = f_i[-75:]
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
# plt.savefig('./scratch/feature_selection_adaboost.png')
# plt.savefig('./figures/feature_selection_adaboost.png')
# plt.show()
plt.close()

first_tuple_elements = []
second_elements = []
for a_tuple in f_i:
    second_elements.append(a_tuple[1])
    first_tuple_elements.append(a_tuple[0])
print('Sum of feature importance', sum(second_elements))

df_features = pd.DataFrame(first_tuple_elements, columns = ["Adaboost_features"])
df_features ["importance"] = second_elements
# print(df_features)
# df_features.to_csv('./data/classifying_data/Adaboost_features.csv', sep = ";", header=True)
# df_features.to_csv('./classifying_data/Adaboost_features.csv', sep = ";", header=True)

def plot_coefficients(classifier, feature_names, top_features=75):
     coef = classifier.feature_importances_
     top_positive_coefficients = np.argsort(coef)[-top_features:]
     top_coefficients = top_positive_coefficients
     
     # create plot
     plt.figure(figsize=(15, 15))
     plt.bar(np.arange(top_features), coef[top_coefficients], color='blue')
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(0, top_features), feature_names[top_coefficients], rotation=40, ha='right')
    #  plt.savefig('./scratch/transposed_feature_selection_adaboost.png')
    #  plt.savefig('./figures/transposed_feature_selection_adaboost.png')
     #  plt.show()
     plt.close()
plot_coefficients(clf, features, 75)


# subset of data frame that only includes the n selected features
first_tuple_elements.append('label')
df_selected = df[first_tuple_elements]

# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df_selected.drop(['label'], axis=1)
y = df_selected.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=20)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=20)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# initialize and train SVM classifier
# clf = AdaBoostClassifier(n_estimators=100, learning_rate = 0.09, random_state=20) # need to adjust according to what the best parameters are
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# save the model to disk
filename = 'FS_adaboost_model.sav'
pickle.dump(fit, open(filename, 'wb')) 

print("########## TEST DATA SET - FEATURE SELECTION ##########")
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
# plt.savefig('./scratch/ROC_adaboost_sel_features.png')
# plt.savefig('./figures/ROC_adaboost_sel_features.png')
plt.close()
# plt.show()

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
# plt.savefig('./scratch/cf_matrix_adaboost_sel_features.png')
# plt.savefig('./figures/cf_matrix_adaboost_sel_features.png')
# plt.show()
plt.close()

# # cf matrix with percentages
# ax= plt.subplot()
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues')
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# # ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# # plt.savefig('./scratch/cf_matrix_perc_adaboost_sel_features.png')
# plt.savefig('./figures/cf_matrix_perc_adaboost_sel_features.png')
# plt.close()
# # plt.show()
