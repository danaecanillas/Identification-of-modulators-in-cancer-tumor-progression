import pandas as pd
import numpy
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

GENERATED_FOLDER = "data/generated/"
DATA_PATH = GENERATED_FOLDER + "clean_train.csv"

data = pd.read_csv(DATA_PATH)

data['class'] = 5
data.loc[(data['RFS']<=730) & (data['RFSE'] == 0), 'class'] = 0 
data.loc[(data['RFS']<=730) & (data['RFSE'] == 1), 'class'] = 1
data.loc[(data['RFS']>1830)  & (data['RFS']<3660) & (data['RFSE'] == 0), 'class'] = 2 
data.loc[(data['RFS']>1830) & (data['RFS']<3660) & (data['RFSE'] == 1), 'class'] = 3 
data = data[data['class'] != 5]

data['PAM50'] = data['PAM50'].astype('category')
data['PAM50'] = data['PAM50'].cat.codes.values

X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != "class"], data["class"], test_size=1/3, random_state=47, stratify=data["class"])

test = pd.concat([X_test, y_test], axis=1)
print("[INFO]: Saving " + GENERATED_FOLDER + "test3.csv")
test.to_csv(GENERATED_FOLDER + "test3.csv",index=False)

print("[INFO]: Target Distribution BEFORE augmentation:")
print(y_train.value_counts())
print("[INFO]: ----------------------------------------\n")
y = y_train
x = X_train.drop(['submitter','INTCLUST'], axis=1)

smote = SMOTE(sampling_strategy = {0: 500, 1: 500, 2:500, 3:500})
x, y = smote.fit_resample(x, y)
result = pd.concat([x, y], axis=1)

print("[INFO]: Target Distribution AFTER augmentation:")
print(result['class'].value_counts())
print("[INFO]: ----------------------------------------")
result.to_csv(GENERATED_FOLDER + "train_augmented3.csv",index=False)