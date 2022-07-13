import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import torch
from torch import nn, optim
from   keras.models import Sequential
from   keras.layers import Dense             # i.e.fully connected

# load data sets
# df_beta_values = pd.read_csv('./data/classifying_data/artistic_study_filt-beta-values_0722_10threshold.txt', sep = ';')
# df_beta_values = pd.read_csv('./data/classifying_data/artistic_study_filt-beta-values_0722_25threshold.txt', sep = ';')
# df_beta_values = pd.read_csv('./data/classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')
df_beta_values = pd.read_csv('./classifying_data/artistic_study_filt-beta-values_0722_50threshold.txt', sep = ';')

# transpose data matrix 
df_beta_transposed = df_beta_values.transpose() 
df_beta_transposed.index.name = 'old_column_name' ## this is to make filtering easier later
df_beta_transposed.reset_index(inplace=True)

# try imputing with several imputation methods
# impute ctrls with ctrls and cases with cases
imputer = SimpleImputer(missing_values = np.nan, strategy ='median')
 
 # extract and add column with labels (0,1) for control and treated samples
df_ctrl = df_beta_transposed.loc[lambda x: x['Phenotype'].str.contains(r'(Control)')]
df_ctrl = df_ctrl.drop(columns =['old_column_name', 'Phenotype'])
imputer1 = imputer.fit(df_ctrl)
imputed_df_ctrl = imputer1.transform(df_ctrl)
df_ctrl_new = pd.DataFrame(imputed_df_ctrl, columns = df_ctrl.columns)
df_ctrl_new.loc[:, 'label'] = 0

df_trt = df_beta_transposed.loc[lambda x: x['Phenotype'].str.contains(r'(Case)')]
df_trt = df_trt.drop(columns =['old_column_name', 'Phenotype'])
imputer2 = imputer.fit(df_trt)
imputed_df_trt = imputer2.transform(df_trt)
df_trt_new = pd.DataFrame(imputed_df_trt, columns = df_trt.columns)
df_trt_new.loc[:, 'label'] = 1

# merge trt and ctrl data frames
df = pd.concat([df_trt_new, df_ctrl_new])
df = df.apply(pd.to_numeric)

# Resampling the minority class. The strategy can be changed as required. (source: https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/)
# sm = SMOTE(sampling_strategy='minority', random_state=42)
# # Fit the model to generate the data.
# oversampled_X, oversampled_Y = sm.fit_resample(df.drop('label', axis=1), df['label'])
# df = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
# df = df.apply(pd.to_numeric)
# print(df.shape)


### Neural Network
# assign X matrix (numeric values to be clustered) and y vector (labels) 
df_X = df.drop(['label'], axis=1)
df_y = df.loc[:, 'label']

# convert to arrays --> nn model requires arrays as inputs
X = df_X.to_numpy()
y = df_y.to_numpy()
features = df_X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
X_train1, X_valid, y_train1, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

# print(df.dtypes)


# parameters for keras
input_dim   = len(X_train.shape[0]) # number of neurons in the input layer
n_neurons   = 50            # number of neurons in the first hidden layer
epochs      = 1000           # number of training cycles


# keras model
model = Sequential()         # a model consisting of successive layers
# input layer
model.add(Dense(n_neurons, input_dim=input_dim, activation='relu'))
# output layer, with one neuron
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=epochs, verbose=0)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)*1


## calculate accuracy
# from sklearn.metrics import confusion_matrix
# with torch.no_grad():
#     model.eval()
#     y_pred = model(torch.tensor(X_test, dtype=torch.float))
#     y_pred_lbl = np.where(y_pred.numpy() > 0, 1, 0)
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
cfm = pd.DataFrame(cf_matrix, columns=["T", "F"], index=["P", "N"])
print(cfm)

# return accuracy and precision score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 Score:", metrics.f1_score(y_test, y_pred))

specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
print('Specificity : ', specificity1 )

sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
print('Sensitivity: ', sensitivity1)

print(metrics.classification_report(y_test, y_pred))

# plot confusion matrix
ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_NN_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_NN_all_features.png')
plt.show()
plt.close()

# cf matrix with percentages
ax= plt.subplot()
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Control', 'Case']); ax.yaxis.set_ticklabels(['Control', 'Case']);
# plt.savefig('./scratch/cf_matrix_percentages_NN_all_features.png')
# plt.savefig('./artistic_trial/plots/cf_matrix_percentages_NN_all_features.png')
plt.show()
plt.close()


# # Deep Explainer for feature selection (source: https://www.kaggle.com/code/ceshine/feature-importance-from-a-pytorch-model/notebook)
# DEVICE = "cpu"
# X_train = X_train.astype(np.float32)

# e = shap.DeepExplainer(
#         model, 
#         torch.from_numpy(
#             X_train[np.random.choice(np.arange(len(X_train)), 300, replace=False)]
#         ).to(DEVICE))

# x_samples = X_train[np.random.choice(np.arange(len(X_train)), 300, replace=False)]
# print(len(x_samples))
# shap_values = e.shap_values(
#     torch.from_numpy(x_samples).to(DEVICE) )
# print(shap_values.shape)

# import pandas as pd
# data = pd.DataFrame({
#     "mean_abs_shap": np.mean(np.abs(shap_values), axis=0), 
#     "stdev_abs_shap": np.std(np.abs(shap_values), axis=0), 
#     "name": features
# })

# data.sort_values("mean_abs_shap", ascending=False)[:10]

# print(data[['mean_abs_shap', 'name']])
# shap.summary_plot(shap_values, features=x_samples, feature_names=features)
# plt.savefig('./scratch/feature_selection_ANN.png')
# # plt.savefig('./artistic_trial/plots/feature_selection_ANN.png')
# # plt.show()
# plt.close()


# ### Neural Network with selected features
# # assign X matrix (numeric values to be clustered) and y vector (labels) 
# features = list(data.name)
# f_i = list(zip(features,data.mean_abs_shap))
# f_i.sort(key = lambda x : x[1])
# f_i = f_i[-50:]
# first_tuple_elements = []
# for a_tuple in f_i:
# 	first_tuple_elements.append(a_tuple[0])
# first_tuple_elements.append('label')

# # subset of data frame that only includes the n selected features
# df_selected = df[first_tuple_elements]
# # assign X matrix (numeric values to be clustered) and y vector (labels) 
# df_X = df_selected.drop(['label'], axis=1)
# df_y = df_selected.loc[:, 'label']

# # convert to arrays --> nn model requires arrays as inputs
# X = df_X.to_numpy()
# y = df_y.to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# X_train1, X_valid, y_train1, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# # print(df.dtypes)
# # hyperparameter optimisation
# # split training set in sub-training sets
# # to find optimal no of layers etc.

# features = df_X.columns
# num_epochs = 5000
# log_inteval = 250
# total_losses = []
# total_val_losses = []
# lr = 1e-4
# lr_decay_inteval = 2500
# lr_decay_rate = 0.3


# model = nn.Sequential(
#     nn.Linear(len(features), 80), # need to find out how and why these params and no of layers + activation function etc
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(80, 256),
#     nn.ReLU(),
#     nn.Dropout(0.6),
#     nn.Linear(256, 1),
# )

# loss_fn = torch.nn.BCELoss()
# opt = optim.Adam(model.parameters(), lr=lr)

# def init_normal(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_normal_(m.weight, 0.06)

# model.apply(init_normal)

# for epoch in range(1, num_epochs+1):
#     y_pred = model(torch.tensor(X_train, dtype=torch.float))
#     y_pred = torch.sigmoid(y_pred)
#     opt.zero_grad()
#     loss = loss_fn(y_pred[:, 0], torch.tensor(y_train, dtype=torch.float))
#     loss.backward()
#     opt.step()
#     total_losses.append(loss.item())
#     if epoch % log_inteval == 0: # Logging
#         epochs_ran = epoch
#         model.eval()
#         with torch.no_grad():
#             y_pred = model(torch.tensor(X_test, dtype=torch.float))
#             y_pred = torch.sigmoid(y_pred)
#             val_loss = loss_fn(y_pred[:, 0], torch.tensor(y_test, dtype=torch.float))
#             total_val_losses.append(val_loss.item())
#         model.train()
#         print(f"total loss in epoch {epoch} = {'%.4f'%loss}, validation loss = {'%.4f'%val_loss}, lr = {'%.2e'%lr}")
#         if len(total_val_losses) > 3 and val_loss.item() > total_val_losses[-2] and val_loss.item() > total_val_losses[-3]:
#             print(f"Validation loss not improving for {log_inteval * 2} epochs, stopping...")
#             break
#     if epoch % lr_decay_inteval == 0: # Learning rate decay
#         lr *= lr_decay_rate
#         for param_group in opt.param_groups:
#             param_group['lr'] = lr

# ## plot
# plt.plot(total_losses, 'b', label="train")
# plt.plot(np.array(range(epochs_ran // log_inteval)) * log_inteval + log_inteval, total_val_losses, 'r', label="valid")
# plt.ylim([0, 1])
# plt.title("Learning curve")
# plt.legend()
# plt.savefig('./scratch/NN_Learning_Curve_selected_features.png')
# # plt.savefig('./artistic_trial/plots/NN_Learning_Curve_selected_features.png')
# plt.close()

# ## calculate accuracy
# # from sklearn.metrics import confusion_matrix
# with torch.no_grad():
#     model.eval()
#     y_pred = model(torch.tensor(X_test, dtype=torch.float))
#     y_pred_lbl = np.where(y_pred.numpy() > 0, 1, 0)
# cf_matrix = metrics.confusion_matrix(y_test, y_pred_lbl)
# cfm = pd.DataFrame(cf_matrix, columns=["T", "F"], index=["P", "N"])
# print(cfm)

# # return accuracy and precision score
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred_lbl))
# print("Precision:", metrics.precision_score(y_test, y_pred_lbl))
# print("Recall:", metrics.recall_score(y_test, y_pred_lbl))
# print("F1 Score:", metrics.f1_score(y_test, y_pred_lbl))

# specificity1 = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
# print('Specificity: ', specificity1 )

# sensitivity1 = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
# print('Sensitivity: ', sensitivity1)

# print(metrics.classification_report(y_test, y_pred_lbl))

# sns.heatmap(cf_matrix, annot=True, fmt='.3g', cmap = 'rocket_r')
# plt.savefig('./scratch/cf_matrix_NN_sel_features.png')
# # plt.savefig('./artistic_trial/plots/cf_matrix_NN_sel_features.png')
# plt.close()
# # plt.show()

# # cf matrix with percentages
# sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
#             fmt='.2%', cmap='Blues')
# plt.savefig('./scratch/cf_matrix_percentages_NN_sel_features.png')
# # plt.savefig('./artistic_trial/plots/cf_matrix_percentages_NN_sel_features.png')
# plt.close()
# # plt.show()


# ## validation set ?


# # lgb_train = lgb.Dataset(X_train, y_train)
# # lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)




