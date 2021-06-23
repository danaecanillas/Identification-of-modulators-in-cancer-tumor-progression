import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import random
import lifelines
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

# Global vars
DATA_PATH = "data/clean_train.csv"

pathways = ['Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2']

cells = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
'T cells CD4 naive', 'T cells CD4 memory resting',
'T cells CD4 memory activated', 'T cells follicular helper',
'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
'Macrophages M2', 'Dendritic cells resting',
'Dendritic cells activated', 'Mast cells resting',
'Mast cells activated', 'Eosinophils', 'Neutrophils']

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

cont_features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8', 
    'T cells CD4 naive', 'T cells CD4 memory resting','T cells CD4 memory activated', 
    'T cells follicular helper', 'T cells regulatory (Tregs)', 'T cells gamma delta','NK cells resting', 
    'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1', 'Macrophages M2','Dendritic cells resting', 
    'Dendritic cells activated', 'Mast cells resting', 'Mast cells activated', 'Eosinophils','Neutrophils', 'Cell_Cycle', 
    'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53', 'WNT', 'Hypoxia','SRC', 'ESR1', 'ERBB2']
      
params = {
    'k_folds': 5,  
    'lr':0.01,
    'epochs':120,
    'batch_size':200,
    'fc1':200,
    'fc2':120,
    'fc3':84,
    'dropout':0.2
}

# Page style
tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'box-shadow': '4px 4px 4px 4px lightgrey',
}
 
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',
}

green_style = {
    'color': 'white',
    'padding': '6px',
    'border-radius': '6px',
    'background-color': '#00CC96',
}

blue_style = {
    'color': 'white',
    'padding': '6px',
    'border-radius': '6px',
    'background-color': '#636EFA',
}

# HIERARCHICAL CLUSTERING

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def heatmap(values):
    data = pd.read_csv('data/clean_train.csv')
    indx = []
    for patient in range(len(data)):
        indx.append(data.iloc[patient]['PAM50'] + " - ")

    id = indx + data['submitter']
    df = data.set_index([id]).sort_index()

    df3 = df[values].transpose()

    fig = go.Figure(data=go.Heatmap(df_to_plotly(df3),colorscale="Plasma"))
    fig.update_layout(
        autosize=False,
        width=1270,
        height=600,
        yaxis=dict(
            title_text="Patient"
        ),
        xaxis=dict(
            title_text="Variable"
        ),
        margin=dict(
            b=100,
            t=50,
        ),
        paper_bgcolor="rgb(247,247,247)",)

    return fig

# TUMOR PREDICTION

def SHAP_VALUES(value):
    df3 = pd.read_pickle("data/df3.pkl")
    Patients = set(df3.index.values)

    res = -1
    while res != 0:
        sample = random.sample(Patients, 1)[0]
        res = sample.find(value)

    Sample = pd.Series(df3.loc[sample])
    col = np.where(Sample<0, '#EF553B', '#00CC96')

    d = {'shap_values': Sample, 'color': col}

    df = pd.DataFrame(data=d)
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

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Net',
            x=features,
            y=df['shap_values'],
            marker_color=df['color']))
    fig.update_layout(
            autosize=False,
            width=1225,
            height=600,
            yaxis=dict(
                title_text="SHAP Value"
            ),
            xaxis=dict(
                title_text="Variable"
            ),
            margin=dict(
                b=100,
                t=50,
            ),
            paper_bgcolor="rgb(247,247,247)",)           
    return fig

# DIMENSIONALITY REDUCTION

def reduce_dimension(method, dimension, x, y, kernel=None):
    if method == "pca":
        pca = PCA(n_components=2)
        X_reduced = pd.DataFrame(pca.fit_transform(x))
        title = "Principal Component Analysis"

    elif method == "tsne":
        X_reduced = pd.DataFrame(TSNE(n_components=2, random_state=6,n_iter=500).fit_transform(x))
        title = "t-distributed Stochastic Neighbor Embedding"
    
    finalDf = pd.concat([y, X_reduced], axis = 1)
    
    fig = px.scatter(x = finalDf.iloc[:,1], y = finalDf.iloc[:,2], hover_data=[finalDf.iloc[:,0]], color = finalDf.iloc[:,0], marginal_x = 'box', marginal_y = 'box', title = title)
    fig.update_layout(
        paper_bgcolor="rgb(247,247,247)")

    return fig

# RELAPSE TIME PREDICTION

def relapse(input1, button2):
      df = pd.read_csv(DATA_PATH)

      kmf = lifelines.KaplanMeierFitter()
      TBasal = df[df.PAM50 == button2]["RFS"]
      EBasal = df[df.PAM50 == button2]["RFSE"]  
      kmf.fit(TBasal, event_observed=EBasal)
      
      return "Result: " + str(round(kmf.predict(int(input1))*100,5)) + "%"

def test(drop11,drop21):
      df = pd.read_csv(DATA_PATH)
      T1 = df[df.PAM50 == drop11]["RFS"]
      E1 = df[df.PAM50 == drop11]["RFSE"]
      T2 = df[df.PAM50 == drop21]["RFS"]
      E2 = df[df.PAM50 == drop21]["RFSE"]
      results = logrank_test(T1,T2, event_observed_A=E1, event_observed_B=E2)

      if results.p_value <= 0.05:
            text = "Significant difference      " + "p-value < 0.05: " + str(results.p_value)
      else:
            text = "No significant difference      " + "p-value > 0.05: " + str(results.p_value)
      return text


    