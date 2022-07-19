import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# load validation data set
df_val = pd.read_csv('./data/classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
df_y_val = pd.read_csv('./data/classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")

X_val = np.array(df_val)
y_val = np.array(df_y_val)

# load machine learning models 
XGB_model = pickle.load(open('XGB_model.sav', 'rb'))
RF_model = pickle.load(open('RF_model.sav', 'rb'))
adaboost_model = pickle.load(open('adaboost_model.sav', 'rb'))
gradBoost_model = pickle.load(open('gradBoost_model.sav', 'rb'))
svm_model = pickle.load(open('svm_model.sav', 'rb'))

print("########## VALIDATION DATA SET ##########")

y_pred = XGB_model.predict(X_val)


# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(XGB_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
# plt.savefig('./artistic_trial/plots/ROC_RF_all_features.png')
plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f75fea)
cf_matrix = metrics.confusion_matrix(y_val, y_pred)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val, y_pred))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_RF_all_features.png')
plt.show()
plt.close()