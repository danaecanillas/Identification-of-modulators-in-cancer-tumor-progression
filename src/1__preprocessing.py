import pandas as pd
import numpy as np
import json
import matplotlib
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
import aux__optimize_neighbors
import os.path
from os import path
import pickle

DATA_PATH = "data/generated/train.csv"
GENERATED_FOLDER = "data/generated/"

df = pd.read_csv(DATA_PATH)

##########################################################
# CODIFICATION
##########################################################

# Change codification of Cellularity
cod = {"CELLULARITY": {"Low": 1, "Moderate": 2, "High":3}}
df = df.replace(cod)

# Change codification of Treatment
def treatment(x):
    ht = 0
    rt = 0
    ct = 0
    Treatment = x.split("/")
    if 'HT' in Treatment:
        ht = 1
    if 'RT' in Treatment:
        rt = 1
    if 'CT' in Treatment:
        ct = 1
    return ht, rt, ct

df["HT"], df["RT"], df["CT"] = zip(*df["Treatment"].map(treatment))
df = df.drop(['Treatment'], axis=1)

# Change codification of Mutations
cod = {"TP53.mut": {"WT": 0, "MUT": 1}}
df = df.replace(cod)
cod = {"PIK3CA.mut": {"WT": 0, "MUT": 1}}
df = df.replace(cod)

##########################################################
# MISSINGS
##########################################################

print(df.isna().sum())

# 5 missings at INTCLUST, remove them
df.drop(df[df["INTCLUST"].isna()].index.values, inplace=True)

# Remove patients with more than 2 variables with Nan
REMOVED_PATIENTS = []
Missings = {}
i = 0
for index, row in df.iterrows():
    nas = row.isna()
    if nas.any():
        missing_cols = nas[nas].index.tolist()

        if (len(missing_cols)) > 2:
            REMOVED_PATIENTS += [row["submitter"]]
            df.drop(index, inplace=True)

        else:
            Missings[row["submitter"]] = missing_cols

print("\nRemoved patients:")
print(REMOVED_PATIENTS)

with open(GENERATED_FOLDER + "info/missings.txt", 'w') as f:  
    f.write(json.dumps(Missings))
f.close()

# Missings input using KNN method 
impute = df.drop(['submitter','PAM50','INTCLUST','RFS','RFSE','DSSE10','DSS10','HT','RT','CT'], axis=1)
print(impute)
target = df['PAM50'].astype('category')
target = target.cat.codes.values

k_file = GENERATED_FOLDER + "K_neighbours.txt"
if(not path.exists(k_file)):
    aux__optimize_neighbors.optimize_k(data=impute, target=target)

file = open(k_file, 'rb')
dict = pickle.load(file)
k = dict["k"]

print("The optimal k is " + str(k))   

imputer = KNNImputer(n_neighbors=k)
imputer.fit(impute)
imputed = pd.DataFrame(imputer.transform(impute), columns=impute.columns, index=impute.index)

df['grade'] = imputed['grade']
df['stage'] = imputed['stage']
df['lymph_nodes_positive'] = imputed['lymph_nodes_positive']
df['CELLULARITY'] = imputed['CELLULARITY']

df.to_csv(GENERATED_FOLDER + "clean_train.csv",index=False)