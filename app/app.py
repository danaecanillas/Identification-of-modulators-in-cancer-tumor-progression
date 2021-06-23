from welcome import welcome_tab
from dimensionality import dim_tab
from modelshap import shap_tab
from kaplan_meier import kaplan_tab
from hierarchical import hierarchical_tab

import lifelines
import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output

import pandas as pd
import json

import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import aux

import torch
import shap
import numpy as np
import pandas as pd
import model.Net as Net

import random


DATA_PATH = "data/clean_train.csv"
MODEL_PATH = 'model/model.pth'

# App settings
app = dash.Dash(__name__,suppress_callback_exceptions=True,meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Identification of modulators in cancer tumor progression"
app.layout =  html.Div([html.Br(),html.Div([html.H1("Identification of modulators in cancer tumor progression")],style={'text-align':'center'}),html.Br(),
    dcc.Tabs(id="tabs-styled-with-inline", value='hello', children=[
        dcc.Tab(label='Welcome', value='hello', style=aux.tab_style, selected_style=aux.tab_selected_style),
        dcc.Tab(label='Hierarchical Clustering', value='heatmap', style=aux.tab_style, selected_style=aux.tab_selected_style),
        dcc.Tab(label='Dimensionality Reduction', value='dim', style=aux.tab_style, selected_style=aux.tab_selected_style),
        dcc.Tab(label='Correlation Study', value='corr', style=aux.tab_style, selected_style=aux.tab_selected_style),
        dcc.Tab(label='Tumor Prediction', value='model1', style=aux.tab_style, selected_style=aux.tab_selected_style),
        dcc.Tab(label='Relapse Probability Prediction', value='model2', style=aux.tab_style, selected_style=aux.tab_selected_style),
    ], style=aux.tabs_styles),
   html.Div(id='tabs-content-inline')
])


#########################################################################
# Dimensionality Reduction
#########################################################################
@app.callback([Output('PCA', 'figure'),Output('TSNE', 'figure')],
              Input('datatable-advanced-filtering', 'derived_virtual_data')
)
def update_graph(derived_virtual_data):
    data = derived_virtual_data
    target = ['PAM50']
    # Separating out the features
    data = pd.DataFrame(data)
        # Separating out the features
    x = data[aux.cont_features].values
    # Separating out the target
    y = data.loc[:,target]

    return aux.reduce_dimension('pca', dimension=2, x=x, y=y, kernel=None), aux.reduce_dimension('tsne', dimension=2, x=x, y=y, kernel=None)

@app.callback(
    Output('datatable-query-structure', 'children'),
    Input('datatable-advanced-filtering', 'derived_filter_query_structure')
)
def display_query(query):
    if query is None:
        return ''
    return html.Details([
        html.Summary('Derived filter query structure'),
        html.Div(dcc.Markdown('''```json
{}
```'''.format(json.dumps(query, indent=4))))
    ])

#########################################################################
# Hierarchical Clustering
#########################################################################
@app.callback(
	Output('pathways_heatmap_output', 'figure'),
	[Input('select_pathways', 'value')])
def update_figure(value):
    if value is None:
        return {'data': []}
    else:  
        return aux.heatmap(value)

@app.callback(
	Output('cells_heatmap_output', 'figure'),
	[Input('select_cells', 'value')])
def update_figure(value):
    if value is None:
        return {'data': []}
    else:  
        return aux.heatmap(value)

#########################################################################
# Tumor Prediction
#########################################################################
@app.callback(Output('SHAP', 'figure'),
	[Input('radio', 'value'),Input('button', 'n_clicks')])
def update_figure(value,n_clicks):
    if n_clicks > 0:
        return aux.SHAP_VALUES(value)
    else:  
        return {'data': []}

@app.callback([Output('button', 'style'),Output('button', 'n_clicks')], [Input('button', 'n_clicks')])
def change_button_style(n_clicks):
    if n_clicks > 0:
        n_clicks = 1
        return aux.blue_style,n_clicks

    else:
        return aux.green_style, n_clicks

#########################################################################
# Relapse Probability Prediction
#########################################################################
@app.callback(
    Output("outputtext", "children"),
    Input("input1", "value"),Input("button2", "value"),
)
def update_output(input1,button2):
    if input1 is None :
        return "Result: 0%"
    
    if input1 == "Time" :
        return "Result: 0%"

    else:
        return aux.relapse()

@app.callback(
    Output("significance", "children"),
    Input("drop11", "value"),Input("drop21", "value"),
)
def update_LogTest(drop11,drop21):
    return aux.test(drop11, drop21)

# MAIN
@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'heatmap':
        return hierarchical_tab()
    elif tab == 'hello':
        return welcome_tab()
    elif tab == 'dim':
        return dim_tab()
    elif tab == 'corr':
        return html.H1("Under Construction 🚧")
    elif tab == 'model1':
        return shap_tab()
    elif tab == 'model2':
        return kaplan_tab()

if __name__ == '__main__':
    app.run_server(debug=True)
