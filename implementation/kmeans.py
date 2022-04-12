import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator


# load data sets
# took example rrbs data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104998
df1 = pd.read_csv('../data/YB5_CRC_study/GSM2813719_YB5_DMSO4d1.txt', sep = '\t')
df2 = pd.read_csv('../data/YB5_CRC_study/GSM2813720_YB5_DMSO4d2.txt', sep = '\t')
df3 = pd.read_csv('../data/YB5_CRC_study/GSM2813721_YB5_DMSO4d3.txt', sep = '\t')

df4 = pd.read_csv('../data/YB5_CRC_study/GSM2813722_YB5_HH1_10uM4d1.txt', sep = '\t')
df5 = pd.read_csv('../data/YB5_CRC_study/GSM2813723_YB5_HH1_10uM4d2.txt', sep = '\t')
df6 = pd.read_csv('../data/YB5_CRC_study/GSM2813724_YB5_HH1_10uM4d3.txt', sep = '\t')

df7 = pd.read_csv('../data/YB5_CRC_study/GSM2813725_YB5_con1.txt', sep = '\t')
df8 = pd.read_csv('../data/YB5_CRC_study/GSM2813726_YB5_con2.txt', sep = '\t')

# add column with labels (0,1) for control and treated samples
# df_trt = pd.concat([df1, df2, df3, df4, df5, df6])
df_trt = df1 # .head(1000)
df_trt['label'] = 1

# df_ctrl = pd.concat([df7, df8])
df_ctrl = df7 # .head(1000)
df_ctrl['label'] = 0

# merge trt and ctrl data frames
df = pd.concat([df_trt, df_ctrl])
print(df.shape)

X = df[['coverage', 'freqC', 'freqT']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)

# The lowest SSE value
kmeans.inertia_

# Final locations of the centroid
kmeans.cluster_centers_

# The number of iterations required to converge
kmeans.n_iter_

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
    
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# kl = KneeLocator(
#     range(1, 11), sse, curve="convex", direction="decreasing"
# )

# kl.elbow

# # A list holds the silhouette coefficients for each k
# silhouette_coefficients = []

# # Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     score = silhouette_score(scaled_features, kmeans.labels_)
#     silhouette_coefficients.append(score)
    
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()