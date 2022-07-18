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

# load training data set
df_train = pd.read_csv('./data/classifying_data/training_data_ARTISTIC_trial.csv', sep = ";")
df_y_train = pd.read_csv('./data/classifying_data/labels_training_data_ARTISTIC_trial.csv', sep = ";")

# load testing data set
df_test = pd.read_csv('./data/classifying_data/testing_data_ARTISTIC_trial.csv', sep = ";")
df_y_test = pd.read_csv('./data/classifying_data/labels_testing_data_ARTISTIC_trial.csv', sep = ";")

# load validation data set
df_val = pd.read_csv('./data/classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
df_y_val = pd.read_csv('./data/classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")

X_train = np.array(df_train)
X_test = np.array(df_test)
X_val = np.array(df_val)

y_train = np.array(df_y_train)
y_test = np.array(df_y_test)
y_val = np.array(df_y_val)

## Hyper Parameter Tuning- finding the best parameters and kernel
## Performance tuning using GridScore
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

# the best params are:
# {'C': 1, 'degree': 1, 'gamma': 0.01, 'kernel': 'linear'}

# using the optimal parameters, initialize and train SVM classifier
clf = svm.SVC(kernel= 'linear', degree = 1, gamma = 0.01, C = 1, 
              class_weight='balanced', probability=True, random_state=20)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

print("########## TEST DATA SET ##########")
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('./scratch/ROC_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/ROC_SVM_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_SVM_all_features.png')
plt.show()
plt.close()

# cf matrix with percentages
ax= plt.subplot()
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
plt.savefig('./scratch/cf_matrix_perc_SVM_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_perc_SVM_all_features.png')
plt.close()
# plt.show()


### feature importances
def plot_coefficients(classifier, feature_names, top_features=75):
     coef = classifier.coef_.ravel()
     print(coef)
     top_positive_coefficients = np.argsort(coef)[-top_features:]
     top_negative_coefficients = np.argsort(coef)[:top_features]
     top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
     
     # create plot
     plt.figure(figsize=(15, 25))
     colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
     plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
     feature_names = np.array(feature_names)
     plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
     plt.savefig('./scratch/transposed_feature_selection_SVM.png')
     plt.show()
     
df = pd.read_csv('./data/classifying_data/complete_data_ARTISTIC_trial.csv', sep = ";")
features = list(df.columns)
plot_coefficients(clf, features, 10)
