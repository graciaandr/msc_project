import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.impute import SimpleImputer
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
imputer = SimpleImputer(missing_values = np.nan, strategy ='constant', fill_value = 5)
 
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


### Neural Network
# assign X matrix (numeric values to be clustered) and y vector (labels) 
df_X = df.drop(['label'], axis=1)
df_y = df.loc[:, 'label']

# convert to arrays --> nn model requires arrays as inputs
X = df_X.to_numpy()
y = df_y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=21)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=80)

print(df.dtypes)

data_cols = df_X.columns
num_epochs = 5000
log_inteval = 250
total_losses = []
total_val_losses = []
lr = 1e-4
lr_decay_inteval = 2500
lr_decay_rate = 0.3

print('hier')

model = nn.Sequential(
    nn.Linear(len(data_cols), 80),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(80, 256),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(256, 1),
)

loss_fn = torch.nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=lr)

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, 0.06)

model.apply(init_normal)

for epoch in range(1, num_epochs+1):
    y_pred = model(torch.tensor(X_train, dtype=torch.float))
    y_pred = torch.sigmoid(y_pred)
    opt.zero_grad()
    loss = loss_fn(y_pred[:, 0], torch.tensor(y_train, dtype=torch.float))
    loss.backward()
    opt.step()
    total_losses.append(loss.item())
    if epoch % log_inteval == 0: # Logging
        epochs_ran = epoch
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test, dtype=torch.float))
            y_pred = torch.sigmoid(y_pred)
            val_loss = loss_fn(y_pred[:, 0], torch.tensor(y_test, dtype=torch.float))
            total_val_losses.append(val_loss.item())
        model.train()
        print(f"total loss in epoch {epoch} = {'%.4f'%loss}, validation loss = {'%.4f'%val_loss}, lr = {'%.2e'%lr}")
        if len(total_val_losses) > 3 and val_loss.item() > total_val_losses[-2] and val_loss.item() > total_val_losses[-3]:
            print(f"Validation loss not improving for {log_inteval * 2} epochs, stopping...")
            break
    if epoch % lr_decay_inteval == 0: # Learning rate decay
        lr *= lr_decay_rate
        for param_group in opt.param_groups:
            param_group['lr'] = lr

## plot
plt.plot(total_losses, 'b', label="train")
plt.plot(np.array(range(epochs_ran // log_inteval)) * log_inteval + log_inteval, total_val_losses, 'r', label="valid")
plt.ylim([0, 1])
plt.title("Learning curve")
plt.legend()
plt.savefig('../scratch/NN_Learning_Curve.png')


## calculate accuracy
from sklearn.metrics import confusion_matrix
with torch.no_grad():
    model.eval()
    y_pred = model(torch.tensor(X_test, dtype=torch.float))
    y_pred_lbl = np.where(y_pred.numpy() > 0, 1, 0)
cm = pd.DataFrame(confusion_matrix(y_test, y_pred_lbl), columns=["T", "F"], index=["P", "N"])
print("Accuracy = %.2f%%" % ((cm.iloc[1, 1] + cm.iloc[0, 0]) / cm.values.sum() * 100))
print(cm)


# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)




