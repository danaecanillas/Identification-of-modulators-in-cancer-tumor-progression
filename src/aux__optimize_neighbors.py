import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))
GENERATED_FOLDER = "data/generated/"

def optimize_k(data, target):
    dict = {}
    errors = []
    for k in range(1, 20, 1):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(data)
        df_imputed = pd.DataFrame(imputed, columns=data.columns)
        
        X = df_imputed
        y = target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append(error)

    ax = sns.lineplot(x=list(map(int, range(1,20))), y=errors)
    ax.set_title('RMSE for different k values')
    fig = ax.get_figure()
    fig.savefig("data/generated/info/K_neighbours.png")

    optim_k = np.argmin(errors) + 1
    dict["k"] = optim_k
    file = open(GENERATED_FOLDER + 'K_neighbours.txt', 'wb')
    pickle.dump(dict, file)
    print("[INFO]: Saving " + GENERATED_FOLDER + 'K_neighbours.txt')
    file.close()
    

