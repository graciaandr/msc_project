import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

# load data sets
df_beta_values = pd.read_csv('../data/classifying_data/CLL_study_filt-beta-values.txt', sep = ';')
# df_beta_values = pd.read_csv('./classifying_data/CLL_study_filt-beta-values.txt', sep = ';')

# transpose data matrix 
df_beta_transposed = df_beta_values.transpose() 
df_beta_transposed.index.name = 'old_column_name' ## this is to make filtering easier later
df_beta_transposed.reset_index(inplace=True)


# try imputing with several imputation methods
# impute ctrls with ctrls and cases with cases
# imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value = 50)
imputer = SimpleImputer(missing_values = np.nan, strategy ='median')

 
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
print(df.shape)

# Resampling the minority class. The strategy can be changed as required. (source: https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/)
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df.drop('label', axis=1), df['label'])
df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
df = df.apply(pd.to_numeric)
print(df.shape)

### Machine Learning 
# assign X matrix (numeric values to be clustered) and y vector (labels) 
X = df.drop(['label'], axis=1)
y = df.loc[:, 'label']

# split data into training and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

### Random Forest Classifier Hyperparameter Tuning
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy
# base_accuracy = evaluate(rf_random, X_test, y_test)

# initialize and train SVM classifier
clf = RandomForestClassifier(max_depth = 30, random_state = 150, n_estimators = 400, min_samples_split = 5, min_samples_leaf = 1,
                             max_features = 'sqrt', bootstrap = True )
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# return accuracy and precision score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))


metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('../scratch/ROC_RF_all_features.png')
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
plt.savefig('../scratch/cf_matrix_RF_all_features.png')
plt.close()
# plt.show()

# cf matrix with percentages
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.savefig('../scratch/cf_matrix_percentages_RF_all_features.png')
plt.close()
# plt.show()

# Feature Selection 
features = list(df.columns)
f_i = list(zip(features,clf.feature_importances_))
f_i.sort(key = lambda x : x[1])
f_i = f_i[-50:]
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.savefig('../scratch/feature_selection_RF.png')
plt.close()
# plt.show()

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# initialize and train SVM classifier
clf = RandomForestClassifier(max_depth=100, random_state=150)
fit = clf.fit(X_train, y_train)

# apply SVM to test data
y_pred = fit.predict(X_test)

# return accuracy and precision score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_test, y_pred))

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.savefig('../scratch/ROC_RF_sel_features.png')
plt.close()
# plt.show()

## calculate and plot confusion matrix (source: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity : ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity (should be same as recall score): ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

sns.heatmap(cf_matrix, annot=True, fmt='.3g')
plt.savefig('../scratch/cf_matrix_RF_sel_features.png')
plt.close()
# plt.show()

# cf matrix with percentages
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
plt.savefig('../scratch/cf_matrix_percentages_RF_sel_features.png')
plt.close()
# plt.show()