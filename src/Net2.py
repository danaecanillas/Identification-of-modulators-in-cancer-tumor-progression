import torch
import random
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

# Set fixed random number seed
torch.manual_seed(31)
torch.cuda.manual_seed(31)
np.random.seed(31)
random.seed(31)

class Net2(nn.Module):

  def __init__(self, n_features, params):

    super(Net2, self).__init__()

    self.fc1 = nn.Linear(n_features, params['fc1'])
    self.fc2 = nn.Linear(params['fc1'], params['fc2'])
    self.fc3 = nn.Linear(params['fc2'], params['fc3'])
    self.fc4 = nn.Linear(params['fc3'], 3)
    self.drop = nn.Dropout(p=params['dropout'])

  def forward(self, x):

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.drop(x)
    x = F.relu(self.fc3(x))
    return self.fc4(x)

def batch_generator(idata, target, batch_size, shuffle=True):
    nsamples = len(idata)
    if shuffle:
        perm = np.random.permutation(nsamples)
    else:
        perm = range(nsamples)

    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if target is not None:
            yield idata[batch_idx], target[batch_idx]
        else:
            yield idata[batch_idx], None

def train(model, criterion, optimizer, X_data, y_data, batch_size, log=False):
    model.train()
    total_loss = 0
    total_acc = 0
    ncorrect = 0
    niterations = 0
    conf = [[0] * 3] * 3

    for X, y in batch_generator(X_data, y_data, batch_size, shuffle=True):
        
        # Get input and target sequences from batch
        X_train = torch.FloatTensor(X)
        y_train = torch.LongTensor(y)
        
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        conf += confusion_matrix(y_train, torch.max(y_pred, 1)[1], labels = [0,1,2,3])    
        
        loss.backward()
        optimizer.step()
        
        # Training statistics
        total_loss += loss.item()
        ncorrect += (torch.max(y_pred, 1)[1] == y_train).sum().item()
        #print(str((torch.max(y_pred, 1)[1] == y_train).sum().item()) +"/"+str(len(X_train)))
        niterations += 1
    
    #print(str(ncorrect) + "/" + str(len(X_data)) + "=" + str(ncorrect/len(X_data)))
    total_acc = ncorrect/len(X_data)*100

    return model, total_loss, total_acc, conf

def val(model, criterion, optimizer, X_data, y_data, batch_size, log=False):
    model.eval()
    total_loss = 0
    total_acc = 0
    ncorrect = 0
    niterations = 0
    conf = [[0] * 4] * 4

    with torch.no_grad():
    
        X_val = torch.FloatTensor(X_data)
        y_val = torch.LongTensor(y_data)

        y_pred = model(X_val)

        conf += confusion_matrix(y_val, torch.max(y_pred, 1)[1], labels = [0,1,2,3])

        ncorrect = (torch.max(y_pred, 1)[1] == y_val).sum().item()
            
    #print(str(ncorrect) + "/" + str(len(X_data)) + "=" + str(ncorrect/len(X_data)))
    total_acc = ncorrect/len(X_data)*100

    return total_acc, conf

