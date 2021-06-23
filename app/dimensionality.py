import dash
import dash_html_components as html
import dash_table

import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import json
import aux

import dash_core_components as dcc

DATA_PATH = "data/clean_train.csv"

def dim_tab():
    df = pd.read_csv(DATA_PATH)
    features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8','T cells CD4 naive',
     'T cells CD4 memory resting','T cells CD4 memory activated', 'T cells follicular helper',
     'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting','NK cells activated',
      'Monocytes', 'Macrophages M0', 'Macrophages M1','Macrophages M2', 'Dendritic cells resting',
      'Dendritic cells activated', 'Mast cells resting','Mast cells activated', 'Eosinophils',
       'Neutrophils', 'Cell_Cycle','HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS',
        'TP53','WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2', 'PROLIF','stage','grade','PAM50']

    return  html.Div([html.Br(),
                html.Div([
                    html.Div([], style={'width':'10%','text-align':'center'}),
                    html.Div([html.H2("PCA"),dcc.Graph(id='PCA')], style={'width':'40%','text-align':'center'}),
                    html.Div([], style={'width':'10%','text-align':'center'}),
                    html.Div([html.H2("TSNE"),dcc.Graph(id='TSNE')], style={'width':'40%','text-align':'center'}),
                    html.Div([], style={'width':'10%','text-align':'center'}),
                    ], style=dict(display='flex')),
                html.Br(),
                html.Hr(),
                html.Div([
                    html.Div([
                        html.Div([], style={'width':'3%','text-align':'center'}),
                        html.Div([html.H3("Filter:"),dash_table.DataTable(
                            id='datatable-advanced-filtering',
                            columns=[
                                {'name': i, 'id': i, 'deletable': False} for i in features
                            ],
                            data=df[features].round(3).to_dict('records'),
                            page_size=10,
                            filter_action="native",
                            style_cell={
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'textAlign': 'left'
                            },
                            style_table={'overflowX': 'auto'}
                        )], style={'width':'73%'}),
                        html.Div([], style={'width':'2%','text-align':'center'})
                    ],style=dict(display='flex')),
                ], style=dict(display='flex')),
                html.Div(id='datatable-query-structure', style={'whitespace': 'pre'}),
            ])
    