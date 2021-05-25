import torch
import numpy as np
import shap
import pandas as pd
import Net

GENERATED_FOLDER = "data/generated/"
DATA_PATH = GENERATED_FOLDER + "train.csv"

def extract_shap(model, features):
    df = pd.read_csv(DATA_PATH)
    df['PAM50'] = df['PAM50'].astype('category')
    df['PAM50'] = df['PAM50'].cat.codes.values

    X = df[features]
    y = df["PAM50"]

    background = X[np.random.choice(X.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(model, torch.FloatTensor(background))

    # shap_values(X, ranked_outputs=None, output_rank_order='max', check_additivity=True)
    # X: A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to explain the model’s output.
    # ranked_outputs: If ranked_outputs is a positive integer then we only explain that many of the top model outputs (where “top” is determined by output_rank_order).

    list_of_tensors = torch.FloatTensor(X)

    shap_values = e.shap_values(list_of_tensors)
    np.save("shap_test.npy", shap_values)