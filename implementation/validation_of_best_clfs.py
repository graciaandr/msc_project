import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# load validation data set
# df_val = pd.read_csv('./data/classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
# df_y_val = pd.read_csv('./data/classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")

df_val = pd.read_csv('./classifying_data/validation_data_ARTISTIC_trial.csv', sep = ";")
df_y_val = pd.read_csv('./classifying_data/labels_validation_data_ARTISTIC_trial.csv', sep = ";")

X_val = np.array(df_val)
y_val = np.array(df_y_val)

# load the top 75 features for each model and see how they perform on those very features 
# feature selection validation sets
# dF_XGB_features = pd.read_csv('./data/classifying_data/XGB_features.csv', sep = ";")
dF_XGB_features = pd.read_csv('./classifying_data/XGB_features.csv', sep = ";")
XGB_features = dF_XGB_features.iloc[:, 1]

# df_RF_features = pd.read_csv('./data/classifying_data/RF_features.csv', sep = ";")
df_RF_features = pd.read_csv('./classifying_data/RF_features.csv', sep = ";")
RF_features = df_RF_features.iloc[:, 1]

# df_Adaboost_features = pd.read_csv('./data/classifying_data/Adaboost_features.csv', sep = ";")
df_Adaboost_features = pd.read_csv('./classifying_data/Adaboost_features.csv', sep = ";")
Adaboost_features = df_Adaboost_features.iloc[:, 1]

# df_GradBoost_features = pd.read_csv('./data/classifying_data/GradBoost_features.csv', sep = ";")
df_GradBoost_features = pd.read_csv('./classifying_data/GradBoost_features.csv', sep = ";")
GradBoost_features = df_GradBoost_features.iloc[:, 1]

# df_SVM_features = pd.read_csv('./data/classifying_data/SVM_features.csv', sep = ";")
df_SVM_features = pd.read_csv('./classifying_data/SVM_features.csv', sep = ";")
SVM_features = df_SVM_features.iloc[:, 1]

# Create the list of sets
# list_of_features = [set(XGB_features), set(RF_features), set(Adaboost_features), set(GradBoost_features), set(SVM_features)]
# list_of_features = [set(RF_features), set(Adaboost_features)]
# list_of_features = [set(RF_features), set(SVM_features)]
# list_of_features = [set(RF_features), set(XGB_features)]
# list_of_features = [set(RF_features), set(GradBoost_features)]
# list_of_features = [set(XGB_features), set(GradBoost_features)]
list_of_features = [set(XGB_features), set(Adaboost_features)]
# list_of_features = [set(XGB_features), set(SVM_features)]
# list_of_features = [set(GradBoost_features), set(Adaboost_features)]
# list_of_features = [set(SVM_features), set(Adaboost_features)]
# list_of_features = [set(SVM_features), set(GradBoost_features)]

intersection = set.intersection(*list_of_features)
print('\n')
print(intersection)
df_intersection = pd.DataFrame(data = intersection, columns=["common_features_XGB_Ada"])
# df_intersection.to_csv('./scratch/common_important_features_XGB_Ada.csv', sep = ";", header=True, index=False)
df_intersection.to_csv('./classifiying_data/common_important_features_XGB_Ada.csv', sep = ";", header=True, index=False)
print(len(intersection))

df_val_XGB = df_val[XGB_features]
df_val_RF = df_val[RF_features]
df_val_Ada = df_val[Adaboost_features]
df_val_Grad = df_val[GradBoost_features]
df_val_SVM = df_val[SVM_features]

X_val_XGB = np.array(df_val_XGB)
X_val_RF = np.array(df_val_RF)
X_val_Ada = np.array(df_val_Ada)
X_val_Grad = np.array(df_val_Grad)
X_val_SVM = np.array(df_val_SVM)

# df_y_val_FS = pd.read_csv('./data/classifying_data/FS_labels_validation_data_ARTISTIC_trial.csv', sep = ";")
df_y_val_FS = pd.read_csv('./classifying_data/FS_labels_validation_data_ARTISTIC_trial.csv', sep = ";")
y_val_FS = np.array(df_y_val_FS)


# load machine learning models 
XGB_model = pickle.load(open('XGB_model.sav', 'rb'))
FS_XGB_model = pickle.load(open('FS_XGB_model.sav', 'rb'))

RF_model = pickle.load(open('RF_model.sav', 'rb'))
FS_RF_model = pickle.load(open('FS_RF_model.sav', 'rb'))

adaboost_model = pickle.load(open('adaboost_model.sav', 'rb'))
FS_adaboost_model = pickle.load(open('FS_adaboost_model.sav', 'rb'))

gradBoost_model = pickle.load(open('gradBoost_model.sav', 'rb'))
FS_gradBoost_model = pickle.load(open('FS_gradBoost_model.sav', 'rb'))

svm_model = pickle.load(open('svm_model.sav', 'rb'))
FS_svm_model = pickle.load(open('FS_svm_model.sav', 'rb'))

print("########## XGBOOST on VALIDATION DATA SET ##########")

y_pred = XGB_model.predict(X_val)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(XGB_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
# plt.savefig('./artistic_trial/plots/ROC_RF_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f75fea)
cf_matrix = metrics.confusion_matrix(y_val, y_pred)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1)

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
plt.savefig('./artistic_trial/plots/cf_matrix_XGB_validation_all_features.png')
# plt.show()
plt.close()

print("########## feature selected XGBOOST on VALIDATION DATA SET ##########")

y_pred2 = FS_XGB_model.predict(X_val_XGB)
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val_FS, y_pred2))
print("Recall:", metrics.recall_score(y_val_FS, y_pred2))
print("F1 Score:", metrics.f1_score(y_val_FS, y_pred2))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val_FS, y_pred2))

metrics.RocCurveDisplay.from_estimator(FS_XGB_model, X_val_XGB, y_val_FS)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_XGB_validation_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix 
cf_matrix = metrics.confusion_matrix(y_val_FS, y_pred2)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val_FS, y_pred2))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
plt.savefig('./artistic_trial/plots/cf_matrix_XGB_validation_sel_features.png')
# plt.show()
plt.close()


print("########## RANDOM FOREST on VALIDATION DATA SET ##########")

y_pred = RF_model.predict(X_val)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(RF_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_RF_validation_all_features.png')
# plt.show()
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
plt.savefig('./artistic_trial/plots/cf_matrix_RF_validation_all_features.png')
# plt.show()
plt.close()

print("########## feature selected RANDOM FOREST on VALIDATION DATA SET ##########")

y_pred2 = FS_RF_model.predict(X_val_RF)
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val_FS, y_pred2))
print("Recall:", metrics.recall_score(y_val_FS, y_pred2))
print("F1 Score:", metrics.f1_score(y_val_FS, y_pred2))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val_FS, y_pred2))

metrics.RocCurveDisplay.from_estimator(FS_RF_model, X_val_RF, y_val_FS)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_RF_validation_sel_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix
cf_matrix = metrics.confusion_matrix(y_val_FS, y_pred2)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val_FS, y_pred2))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
plt.savefig('./artistic_trial/plots/cf_matrix_RF_validation_sel_features.png')
# plt.show()
plt.close()

print("########## ADABOOST on VALIDATION DATA SET ##########")

y_pred = adaboost_model.predict(X_val)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(adaboost_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_ada_validation_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix
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
plt.savefig('./artistic_trial/plots/cf_matrix_ada_validation_all_features.png')
# plt.show()
plt.close()

print("########## feature selected ADABOOST on VALIDATION DATA SET ##########")

y_pred2 = FS_adaboost_model.predict(X_val_Ada)
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val_FS, y_pred2))
print("Recall:", metrics.recall_score(y_val_FS, y_pred2))
print("F1 Score:", metrics.f1_score(y_val_FS, y_pred2))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val_FS, y_pred2))

metrics.RocCurveDisplay.from_estimator(FS_adaboost_model, X_val_Ada, y_val_FS)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_ada_validation_sel_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f75fea)
cf_matrix = metrics.confusion_matrix(y_val_FS, y_pred2)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val_FS, y_pred2))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
plt.savefig('./artistic_trial/plots/cf_matrix_ada_validation_sell_features.png')
# plt.show()
plt.close()

print("########## GRADBOOST on VALIDATION DATA SET ##########")

y_pred = gradBoost_model.predict(X_val)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(gradBoost_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_grad_validation_all_features.png')
# plt.show()
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
plt.savefig('./artistic_trial/plots/cf_matrix_grad_validation_all_features.png')
# plt.show()
plt.close()

print("########## feature selected GRADBOOST on VALIDATION DATA SET ##########")

y_pred2 = FS_gradBoost_model.predict(X_val_Grad)
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val_FS, y_pred2))
print("Recall:", metrics.recall_score(y_val_FS, y_pred2))
print("F1 Score:", metrics.f1_score(y_val_FS, y_pred2))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val_FS, y_pred2))

metrics.RocCurveDisplay.from_estimator(FS_gradBoost_model, X_val_Grad, y_val_FS)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_grad_validation_all_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f75fea)
cf_matrix = metrics.confusion_matrix(y_val_FS, y_pred2)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val, y_pred2))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
plt.savefig('./artistic_trial/plots/cf_matrix_grad_validation_sel_features.png')
# plt.show()
plt.close()

print("########## SVM on VALIDATION DATA SET ##########")

y_pred = svm_model.predict(X_val)

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
print("Recall:", metrics.recall_score(y_val, y_pred))
print("F1 Score:", metrics.f1_score(y_val, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val, y_pred))

metrics.RocCurveDisplay.from_estimator(svm_model, X_val, y_val)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_SVM_validation_all_features.png')
# plt.show()
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
plt.savefig('./artistic_trial/plots/cf_matrix_SVM_validation_all_features.png')
# plt.show()
plt.close()

print("########## feature selected SVM on VALIDATION DATA SET ##########")

y_pred2 = FS_svm_model.predict(X_val_SVM)
# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_val_FS, y_pred2))
print("Recall:", metrics.recall_score(y_val_FS, y_pred2))
print("F1 Score:", metrics.f1_score(y_val_FS, y_pred2))
print("AUC-ROC Score:", metrics.roc_auc_score(y_val_FS, y_pred2))

metrics.RocCurveDisplay.from_estimator(FS_svm_model, X_val_SVM, y_val_FS)
# plt.savefig('../scratch/ROC_RF_all_features.png')
plt.savefig('./artistic_trial/plots/ROC_SVM_validation_sel_features.png')
# plt.show()
plt.close()

# calculate and plot confusion matrix 
cf_matrix = metrics.confusion_matrix(y_val_FS, y_pred2)

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_val_FS, y_pred2))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_RF_all_features.png')
plt.savefig('./artistic_trial/plots/cf_matrix_SVM_validation_sel_features.png')
# plt.show()
plt.close()