import pandas as pd
import numpy
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

GENERATED_FOLDER = "data/generated/"
DATA_PATH = GENERATED_FOLDER + "clean_train.csv"

df = pd.read_csv(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != "PAM50"], df["PAM50"], test_size=1/3, random_state=47, stratify=df["PAM50"])
 
test = pd.concat([X_test, y_test], axis=1)
print("[INFO]: Saving " + GENERATED_FOLDER + "test.csv")
test.to_csv(GENERATED_FOLDER + "test.csv",index=False)

train = pd.concat([X_train, y_train], axis=1)
print("[INFO]: Saving " + GENERATED_FOLDER + "train.csv")
test.to_csv(GENERATED_FOLDER + "train.csv",index=False)

print("\n[INFO]: Target Distribution BEFORE augmentation:")
print(y_train.value_counts())
print("[INFO]: ----------------------------------------\n")
y = y_train
x = X_train.drop(['submitter','INTCLUST'], axis=1)

smote = SMOTE(sampling_strategy = {"LumA": 470, "LumB": 470, "Basal": 470, "Her2": 470})
x, y = smote.fit_resample(x, y)
result = pd.concat([x, y], axis=1)

print("[INFO]: Target Distribution AFTER augmentation:")
print(result['PAM50'].value_counts())
print("[INFO]: ----------------------------------------")
result.to_csv(GENERATED_FOLDER + "learn_augmented.csv",index=False)