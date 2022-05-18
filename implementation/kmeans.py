import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from regex import P
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator
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
imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value=5)
 
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
labels = df.loc[:,'label']
data = df.drop('label', axis=1, inplace=False)
oversampled_X, oversampled_Y = sm.fit_resample(data, labels)
df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
df = df.apply(pd.to_numeric)

kmeans = KMeans(
    init="random",
    n_clusters=2,
    n_init=15,
    max_iter=1000,
    random_state=42
)

# kmeans.fit(scaled_features)
clusters = kmeans.fit_predict(df)
df['cluster'] = clusters
print('clustering done')
print(df.head(5))

y_pred = df['cluster']
y_real = df['label']
cf_matrix = metrics.confusion_matrix(y_real, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.3g')
plt.show()

sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

# return evaluation metrics
print("Accuracy:", metrics.accuracy_score(y_real, y_pred))
print("Recall:", metrics.recall_score(y_real, y_pred))
print("F1 Score:", metrics.f1_score(y_real, y_pred))
print("AUC-ROC Score:", metrics.roc_auc_score(y_real, y_pred))
print(metrics.classification_report(y_real, y_pred))

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity: ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)
