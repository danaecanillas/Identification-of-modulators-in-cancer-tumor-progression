import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from aux import features, params
import model.Net as Net

import torch
import shap
import numpy as np
import pandas as pd


DATA_PATH = "data/clean_train.csv"
MODEL_PATH = 'model/model.pth'

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

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

def heatmap2():
    # Load the data
    df3 = pd.read_pickle("data/df3.pkl")
    df3[df3 < 1] = df3[df3 < 1].transform(lambda x: -np.log(-x + 1))
    df3[df3 > 1] = df3[df3 > 1].transform(lambda x: np.log(x + 1))
    df3 = df3.transpose()

    fig = go.Figure(data=go.Heatmap(df_to_plotly(df3),colorscale="Plasma"))
    fig.update_layout(
        autosize=False,
        width=1400,
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
    
def shap_tab():

    return  html.Div(
                [
                html.Br(),
                html.Div([
                    html.Div([], style={'width':'10%','text-align':'center'}),
                    html.Div([html.H2("Saliency Values"),dcc.Graph(figure=heatmap2())], style={'width':'40%'}),
                    html.Div([], style={'width':'10%','text-align':'center'})
                    ], style=dict(display='flex')),
                html.Br(),html.Br(),
                html.Div([
                    html.Div([
                        html.Div([], style={'width':'10%','text-align':'center'}),
                            html.Div([
                                html.H2("Shapley Values"),
                                html.P("Select a tumor type:"),
                                html.Div([
                                    html.Div([  dcc.RadioItems(
                                                    id='radio',
                                                    options=[
                                                        {'label': 'Basal', 'value': 'Basal'},
                                                        {'label': 'Her2', 'value': 'Her2'},
                                                        {'label': 'LumA', 'value': 'LumA'},
                                                        {'label': 'LumB', 'value': 'LumB'}
                                                    ],
                                                    value='Basal',
                                                    labelStyle={'display': 'block'}),
                                                html.Br(),
                                                html.Button(
                                                    id='button',
                                                    children=['Random Trial'],
                                                    n_clicks=1,
                                                    style=blue_style)], style={'width':'40%'}),
                                                html.Div([], style={'width':'10%','text-align':'center'}),
                                                html.Div([html.Div([html.Div([dcc.Graph(id='SHAP')])
                                                                    ]
                                                        ,style=dict(display='flex')),], style={'width':'80%','text-align':'center'})
                                                ], 
                                    style=dict(display='flex')),], style={'width':'50%'}),
                                
                                html.Div([], style={'width':'10%','text-align':'center'})
                            ], style=dict(display='flex')),
                    ],)
                ])
            