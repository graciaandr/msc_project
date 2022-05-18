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
labels = df.loc[:,'label']
print('hier1')

data = df.drop('label', axis=1, inplace=False)

print(data.columns)
print(labels)
print('hier2')
oversampled_X, oversampled_Y = sm.fit_resample(data, labels)
df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
df = df.apply(pd.to_numeric)
print(df.shape)


# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])
df.drop('label', axis=1)
print(df.shape)

stop1

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

pred_label = df['cluster']
real_label = df['label']
cf_matrix = metrics.confusion_matrix(real_label, pred_label)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()

mymap = {0:'.', 1:'s'}
df['cluster'].map(lambda s: mymap.get(s) if s in mymap else s)
print(df.head(5))

#plotting the results
plt.scatter(df.freqC, df.freqT, c=df.cluster, alpha = 0.6, s = df.coverage)
plt.show()


# # The lowest SSE value
# print(kmeans.inertia_)

# # Final locations of the centroid
# print(kmeans.cluster_centers_)

# The number of iterations required to converge
# print('n_iter: ', kmeans.n_iter_)

# kmeans_kwargs = {
#     "init": "random",
#     "n_init": 10,
#     "max_iter": 1000,
#     "random_state": 42,
# }

# A list holds the SSE values for each k
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(df)
#     sse.append(kmeans.inertia_)
    
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()

# kl = KneeLocator(
#     range(1, 11), sse, curve="convex", direction="decreasing"
# )

# print('elbow: ', kl.elbow)

# A list holds the silhouette coefficients for each k
# silhouette_coefficients = []

# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(df)
#     score = silhouette_score(df, kmeans.labels_)
#     silhouette_coefficients.append(score)
    
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()