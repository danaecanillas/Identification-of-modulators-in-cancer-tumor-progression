import pandas as pd
import numpy
from imblearn.over_sampling import SMOTE

GENERATED_FOLDER = "data/generated/"
DATA_PATH = GENERATED_FOLDER + "clean_train.csv"

df = pd.read_csv(DATA_PATH)

print("[INFO]: Target Distribution BEFORE augmentation:")
print(df['PAM50'].value_counts())
print("[INFO]: ----------------------------------------\n")
y = df['PAM50']
x = df.drop(['submitter','PAM50','INTCLUST'], axis=1)

smote = SMOTE(sampling_strategy = {"LumA": 706, "LumB": 700, "Basal": 700, "Her2": 700})
x, y = smote.fit_resample(x, y)
result = pd.concat([x, y], axis=1)

print("[INFO]: Target Distribution AFTER augmentation:")
print(result['PAM50'].value_counts())
print("[INFO]: ----------------------------------------")
result.to_csv(GENERATED_FOLDER + "train_augmented.csv",index=False)