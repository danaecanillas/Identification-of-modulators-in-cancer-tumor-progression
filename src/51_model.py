import os
import numpy as np
import time
import random
import torch
import shap
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import Net
from confusion import cm
import matplotlib.pyplot as plt


# Set fixed random number seed
torch.manual_seed(31)
torch.cuda.manual_seed(31)
np.random.seed(31)
random.seed(31)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()

def get_data(features):
  learn_data = "data/generated/train_augmented3.csv"
  learn_data = pd.read_csv(learn_data)
  learn_data['class'] = learn_data['class'].astype('category')
  learn_data['class'] = learn_data['class'].cat.codes.values
  
  X_learn = learn_data[features]
  y_learn = learn_data["class"]

  test_data = "data/generated/test3.csv"
  test_data = pd.read_csv(test_data)
  test_data['class'] = test_data['class'].astype('category')
  test_data['class'] = test_data['class'].cat.codes.values
  
  X_test = test_data[features]
  y_test = test_data["class"]
  
  return X_learn, y_learn, X_test, y_test

if __name__ == '__main__':
  features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
      'T cells CD4 naive', 'T cells CD4 memory resting',
      'T cells CD4 memory activated', 'T cells follicular helper',
      'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
      'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
      'Macrophages M2', 'Dendritic cells resting',
      'Dendritic cells activated', 'Mast cells resting',
      'Mast cells activated', 'Eosinophils', 'Neutrophils', 'Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2', 'PROLIF']
  
  params = {
    'k_folds': 5,  
    'lr':0.001,
    'epochs':120,
    'batch_size':200,
    'fc1':200,
    'fc2':120,
    'fc3':84,
    'dropout':0.1
}
  # Configuration options
  k_folds = params['k_folds']
  num_epochs = params['epochs']
  loss_function = nn.CrossEntropyLoss()
  
  # For fold results
  results = {}
  
  # Prepare MNIST dataset by concatenating Train/Test part; we split later.
  X_learn, y_learn, X_test, y_test = get_data(features)
  
  X_learn = X_learn.to_numpy(dtype=float)
  y_learn = y_learn.to_numpy()

  X_test = X_test.to_numpy(dtype=float)
  y_test = y_test.to_numpy()

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
    
  # Start printoutput_rank_order
  # K-fold Cross Validation model evaluation
  for fold, (train_ids, val_ids) in enumerate(kfold.split(X_learn,y_learn),1):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    X_train_fold = X_learn[train_ids] 
    y_train_fold = y_learn[train_ids]

    X_val_fold = X_learn[val_ids] 
    y_val_fold = y_learn[val_ids]

    n_features = len(X_train_fold[0])

    # Init the neural network
    model = Net.Net(n_features, params)
    model.apply(reset_weights)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    start_time = time.time()
    epochs = params['epochs']
    loss_ = []
    train_acc = []
    val_acc = []
    epochs_ = []

    # Run the training loop for defined number of epochs
    for epoch in range(1, num_epochs+1):
      epochs_.append(epoch)
      model, loss, t_acc, t_conf = Net.train(model, loss_function, optimizer, X_train_fold, y_train_fold, batch_size=params['batch_size'])
      train_acc.append(t_acc)
      loss_.append(loss)

      v_acc, v_conf = Net.val(model, loss_function, optimizer, X_val_fold, y_val_fold, batch_size=params['batch_size'])
      val_acc.append(v_acc)
      if epoch % 20 == 0:
        print(f'| epoch {epoch:03d} | loss={round(loss, 4):.3n}')
        print(f"--- %s seconds --- {round((time.time() - start_time),4):.3n}")
        print(f'    - Training accuracy   = {round(t_acc,4):.3n}%')
        print(f'    - Validation accuracy = {round(v_acc,4):.3n}% \n')

    print('Training process has finished.')
    # Saving the model
    print('-> Saving trained model: /model/model-fold-'+str(fold)+'.pth')
    save_path = f'./model/model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Print accuracy
    print('\nAccuracy for fold %d: %f %% \n' % (fold, v_acc))
    print('--------------------------------')
    results[fold] = v_acc

  # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'- Fold {key}: {value} %')
    sum += value
  print('--------------------------------')
  print(f'Average: {sum/len(results.items())} %')
  print('--------------------------------')

  # TRAINING
  model = Net.Net(n_features, params)
  model.apply(reset_weights)

  # Initialize optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

  start_time = time.time()
  epochs = params['epochs']
  loss_ = []
  train_acc = []
  val_acc = []
  epochs_ = []

  # Run the training loop for defined number of epochs
  for epoch in range(1, num_epochs+1):
    epochs_.append(epoch)
    model, loss, t_acc, t_conf = Net.train(model, loss_function, optimizer, X_learn, y_learn, batch_size=params['batch_size'])
    train_acc.append(t_acc)
    loss_.append(loss)

    v_acc, v_conf = Net.val(model, loss_function, optimizer, X_test, y_test, batch_size=params['batch_size'])
    val_acc.append(v_acc)
    if epoch % 20 == 0:
      print(f'| epoch {epoch:03d} | loss={round(loss, 4):.3n}')
      print(f"--- %s seconds --- {round((time.time() - start_time),4):.3n}")
      print(f'    - Training accuracy   = {round(t_acc,4):.3n}%')
      print(f'    - Test accuracy = {round(v_acc,4):.3n}% \n')

  cm(v_conf)



