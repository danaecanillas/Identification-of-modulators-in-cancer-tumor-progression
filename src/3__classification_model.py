import os
import numpy as np
import time
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import Net
from confusion_matrix_generator import cm
import matplotlib.pyplot as plt

# Set fixed random number seed
torch.manual_seed(31)
torch.cuda.manual_seed(31)
np.random.seed(31)
random.seed(31)

def reset_weights(m):
  '''
    Reset model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()


def get_data(features):
  '''
  Function that obtains the augmented data and the test data.
  '''
  learn_data = "data/generated/train_augmented.csv"
  learn_data = pd.read_csv(learn_data)

  # Encode the target variable numerically
  learn_data['PAM50'] = learn_data['PAM50'].astype('category')
  learn_data['PAM50'] = learn_data['PAM50'].cat.codes.values
  
  X_learn = learn_data[features]
  y_learn = learn_data["PAM50"]

  result = X_learn.copy()

  # Data normalization
  result['stage']=(result['stage']-result['stage'].min())/(result['stage'].max()-result['stage'].min())
  result['grade']=(result['grade']-result['grade'].min())/(result['grade'].max()-result['grade'].min())
  #result['age_at_diagnosis']=(result['age_at_diagnosis']-result['age_at_diagnosis'].min())/(result['age_at_diagnosis'].max()-result['age_at_diagnosis'].min())

  X_learn = result

  test_data = "data/generated/test.csv"
  test_data = pd.read_csv(test_data)
  test_data['PAM50'] = test_data['PAM50'].astype('category')
  test_data['PAM50'] = test_data['PAM50'].cat.codes.values

  result = test_data.copy()

  # Data normalization
  result['stage']=(result['stage']-result['stage'].min())/(result['stage'].max()-result['stage'].min())
  result['grade']=(result['grade']-result['grade'].min())/(result['grade'].max()-result['grade'].min())
  #result['age_at_diagnosis']=(result['age_at_diagnosis']-result['age_at_diagnosis'].min())/(result['age_at_diagnosis'].max()-result['age_at_diagnosis'].min())
  
  test_data = result

  X_test = test_data[features]
  y_test = test_data["PAM50"]
  
  return X_learn, y_learn, X_test, y_test


def Kfold(k_folds, X_learn, y_learn, num_epochs, loss_function):
  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)
  
  # For fold results
  results = {}
    
  # K-fold Cross Validation model evaluation
  ############################################################################
  for fold, (train_ids, val_ids) in enumerate(kfold.split(X_learn,y_learn),1):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Split into folds
    X_train_fold = X_learn[train_ids] 
    y_train_fold = y_learn[train_ids]

    X_val_fold = X_learn[val_ids] 
    y_val_fold = y_learn[val_ids]

    # Number of variables
    n_features = len(X_train_fold[0])

    # Init the neural network
    model = Net.Net(n_features, params)
    model.apply(reset_weights)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], eps=params['eps'], weight_decay=params['weight_decay'])

    start_time = time.time()
    loss_ = []
    train_acc = []
    val_acc = []
    epochs_ = []

    # Run the training loop for defined number of epochs
    for epoch in range(1, num_epochs+1):

      # Training
      epochs_.append(epoch)
      model, loss, t_acc, t_conf = Net.train(model, loss_function, optimizer, X_train_fold, y_train_fold, batch_size=params['batch_size'])
      train_acc.append(t_acc)
      loss_.append(loss)

      # Validation
      v_acc, v_conf, _ = Net.val(model, loss_function, optimizer, X_val_fold, y_val_fold, batch_size=params['batch_size'])
      val_acc.append(v_acc)

      # Print the results
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

  val_avg = sum/len(results.items())
  print('--------------------------------')
  print(f'Average: {val_avg} %')
  print('--------------------------------')

  return val_avg

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['Acc'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    d = np.diag(df_cm) 
    d = np.append (d, np.sum(np.diag(df_cm)))
    df_cm.loc['Accuracy'] = d / sum_col
    return df_cm

def TRAIN(params):
  # Configuration options
  cross_validation = True
  k_folds = 5
  num_epochs = params['epochs']
  loss_function = nn.CrossEntropyLoss()
  
  # Prepare MNIST dataset by concatenating Train/Test part; we split later.
  X_learn, y_learn, X_test, y_test = get_data(features)
  
  X_learn = X_learn.to_numpy(dtype=float)
  y_learn = y_learn.to_numpy()

  X_test = X_test.to_numpy(dtype=float)
  y_test = y_test.to_numpy()

  if cross_validation:
    val = Kfold(k_folds, X_learn,y_learn, num_epochs, loss_function)

  # TRAINING (with the entire data)
  #######################################################NeuralNeuralNeuralNeural#####################
  n_features = len(X_learn[0])
  model = Net.Net(n_features, params)
  model.apply(reset_weights)

  # Initialize optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

  start_time = time.time()
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

    if epoch % 20 == 0:
      print(f'| epoch {epoch:03d} | loss={round(loss, 4):.3n}')
      print(f"--- %s seconds --- {round((time.time() - start_time),4):.3n}")
      print(f'    - Training accuracy   = {round(t_acc,4):.3n}%')    

  # Save the model
  print('-> Saving trained model: /model/model.pth')
  save_path = f'./model/model.pth'
  torch.save(model.state_dict(), save_path)

  # Validation
  v_acc, v_conf, y_pred = Net.val(model, loss_function, optimizer, X_test, y_test, batch_size=params['batch_size'])
  val_acc.append(v_acc)
  print(f'Test accuracy = {round(v_acc,4):.3n}% \n')

  # Print confussion matrix
  #cm(a)
  df_cm = pd.DataFrame(v_conf, index=["Basal", "Her2", "LumA", "LumB"], columns=["Basal", "Her2", "LumA", "LumB"])
  df = insert_totals(df_cm)  

  d = {'lr': params['lr'], 'epochs': params['epochs'], 'batch_size': params['batch_size'], 
  'fc1': params['fc1'], 'fc2': params['fc2'], 'fc3': params['fc3'] , 'dropout': params['dropout'], 'eps': params['eps'], 'weight_decay': params['weight_decay'],
  'Tr': round(t_acc,4), 'Val': round(val,4), 'Te': round(v_acc,4), 'Basal_val' : round(np.array(df.tail(1))[0][0]*100,4), 'Her2_val' : round(np.array(df.tail(1))[0][1]*100,4), 'LumA_val' : round(np.array(df.tail(1))[0][2]*100,4), 'LumB_val' : round(np.array(df.tail(1))[0][3]*100,4)}

  df = pd.DataFrame.from_records([(d)])
  return df

if __name__ == '__main__':

  d = {'lr': 0, 'epochs': 0, 'batch_size': 0, 
  'fc1': 0, 'fc2': 0, 'fc3': 0, 'dropout': 0, 'eps': 0, 'weight_decay': 0,
  'Tr': 0, 'Val': 0, 'Te': 0, 'Basal_val' : 0, 'Her2_val' : 0, 'LumA_val' : 0, 'LumB_val' : 0}

  df = pd.DataFrame.from_records([(d)])

  features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
      'T cells CD4 naive', 'T cells CD4 memory resting',
      'T cells CD4 memory activated', 'T cells follicular helper',
      'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
      'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
      'Macrophages M2', 'Dendritic cells resting',
      'Dendritic cells activated', 'Mast cells resting',
      'Mast cells activated', 'Eosinophils', 'Neutrophils', 'Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2', 'PROLIF','stage','grade']
  
  hyper_lr = [0.01, 0.001]
  hyper_epochs = [80, 100, 120]
  hyper_batch_size = [100, 150, 200]
  hyper_net = [[40, 20, 10]]
  hyper_dropout = [0.1, 0.2]
  hyper_eps = [1e-8, 1e-6]
  hyper_weight_decay = [1e-4, 1e-2]

  #hyper_lr = [0.001]
  #hyper_epochs = [80]
  #hyper_batch_size = [200]
  #hyper_net = [[40, 20, 10]]
  #hyper_dropout = [0.2]
  #hyper_eps = [1e-08]
  #hyper_weight_decay = [0.0001]

  for lr in hyper_lr:
          for eps in hyper_eps:
              for weight in hyper_weight_decay:
                  for epochs in hyper_epochs:
                      for batch_size in hyper_batch_size:
                          for fc in hyper_net:
                              for dropout in hyper_dropout:
                                  params = {
                                          'lr':lr,
                                          'eps':eps,
                                          'weight_decay':weight,
                                          'epochs':epochs,
                                          'batch_size':batch_size,
                                          'fc1':fc[0],
                                          'fc2':fc[1],
                                          'fc3':fc[2],
                                          'dropout':dropout,
                                          'features':features
                                          }
                                  res = TRAIN(params)
                                  df = df.append(res, ignore_index=True)

  df.to_csv("model1.csv",index=False, sep=";")




