import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

# Step1: Reading Data
dataset_path = "/DESKTOP/tess20222XXXXXX_dvr-tcestats.csv";
dfc = pd.read_csv(dataset_path)

koi_disposition_labels = {
    "koi_disposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    },
    "koi_pdisposition": {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "CANDIDATE": 2,
        "NOT DISPOSITIONED": 3
    }
}

df_multi.replace(koi_disposition_labels, inplace=True)
df_multi

# Step2: Making the train data

df_multi = df_multi.select_dtypes(exclude=['object']).copy()
df_test = df_multi.copy()
cols_to_be_removed = ['kepid', 'koi_pdisposition', 'koi_score', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_teq_err1', 'koi_teq_err2']
df_multi.drop(cols_to_be_removed, axis=1, inplace=True)
test_data_cols_to_be_removed = [col for col in cols_to_be_removed if col not in ['koi_pdisposition', 'koi_score']]
df_test.drop(test_data_cols_to_be_removed, axis=1, inplace=True)
df_multi = df_multi.fillna(0)

index = df_multi[df_multi.koi_fpflag_nt == df_multi.koi_fpflag_nt.max()].index
df_multi.drop(index, inplace=True)
df_multi.info()

#Step3: Making the test data
df_test = df_test[df_test.koi_disposition == 2]
df_standarized = df_multi.copy()
std_scaler = StandardScaler()
df_standarized.iloc[:, 5:] = std_scaler.fit_transform(df_standarized.iloc[:, 5:])

#Step4: Making the Model
class ExoplanetDataset(Dataset):
    def __init__(self, test=False):
        self.dataframe_orig = dfc = pd.read_csv(dataset_path, skiprows = 144)

        if (test == False):
            self.data = df_standarized[( df_standarized.koi_disposition == 1 ) | ( df_standarized.koi_disposition == 0 )].values
        else:
            self.data = df_standarized[~(( df_standarized.koi_disposition == 1 ) | ( df_standarized.koi_disposition == 0 ))].values
            
        self.X_data = torch.FloatTensor(self.data[:, 1:])
        self.y_data = torch.FloatTensor(self.data[:, 0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    
    def get_col_len(self):
        return self.X_data.shape[1]
    
exoplanet_df = ExoplanetDataset()

#Step5: splitting the data
torch.manual_seed(42)
batch_size = 32
n_epochs = 1000

first_set = int(len(exoplanet_df) * 0.8)
test_size = len(exoplanet_df) - first_set
train_size = first_set * 0.8
val_size = first_set - train_size

first_set_ds, test_ds = random_split(exoplanet_df, [first_set, test_size])
train_ds, val_ds = random_split(exoplanet_df, [first_set, test_size])
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

for features, target in train_loader:
    print(features.size(), target.size())
    break

class EXClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(EXClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)    
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, out_dim)
         
    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)
        out = self.linear5(out)
        out = torch.sigmoid(out)
        return out
    
    
    def predict(self, x):
        predict = self.forward(x)
        return predict
        
    def print_params(self):
        for params in self.parameters():
            print(params)

input_dim = exoplanet_df.get_col_len()
out_dim = 1
model = EXClassifier(input_dim, out_dim)

#Step6: training the model
criterion = nn.BCELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

def train_model():
    for X, y in train_loader:
        for epoch in range(n_epochs):
            optim.zero_grad()
            y_pred = model.forward(X).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()

train_model()

#Step7: displaying evaluative metrics
from sklearn.metrics import confusion_matrix
def pred_confusion_matrix(model, loader):
    with torch.no_grad():
        all_preds = torch.tensor([])
        all_true = torch.tensor([])
        for X, y in loader:
            y_pred = model(X)
            y_pred = torch.tensor(y_pred > 0.5, dtype=torch.float32).flatten()
            all_preds = torch.cat([all_preds, y_pred])
            all_true = torch.cat([all_true, y])
            
    
    return confusion_matrix(all_true.numpy(), all_preds.numpy())

#Optional: in case you want to see the metrics.
%matplotlib inline #if run in a jupter notebook

confusion_matrix_training   = pred_confusion_matrix(model, train_loader)
confusion_matrix_validation = pred_confusion_matrix(model, val_loader)
confusion_matrix_testing    = pred_confusion_matrix(model, test_loader)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax1, ax2, ax3 = axes
sns.heatmap(confusion_matrix_training, fmt='g', annot=True, ax=ax1)
ax1.set_title('Training Data')

sns.heatmap(confusion_matrix_validation, fmt='g', annot=True, ax=ax2)
ax2.set_title('Validation Data')

sns.heatmap(confusion_matrix_testing, fmt='g', annot=True, ax=ax3)
ax3.set_title('Testing Data')

save = {'model_state': model.state_dict(), 'optimizer_state': optim.state_dict()}
torch.save(save, 'model_saving.pth')