from welcome import welcome_tab
from dimensionality import dim_tab
from modelshap import shap_tab
from kaplan_meier import kaplan_tab

from lifelines.statistics import logrank_test

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


app = dash.Dash(__name__,suppress_callback_exceptions=True,meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

app.title = "Identification of modulators in cancer tumor progression"


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


app.layout =  html.Div([html.Br(),html.Div([html.H1("Identification of modulators in cancer tumor progression")],style={'text-align':'center'}),html.Br(),
    dcc.Tabs(id="tabs-styled-with-inline", value='hello', children=[
        dcc.Tab(label='Welcome', value='hello', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Hierarchical Clustering', value='heatmap', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Dimensionality Reduction', value='dim', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Correlation Study', value='corr', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Tumor Prediction', value='model1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Relapse Probability Prediction', value='model2', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
   html.Div(id='tabs-content-inline')
])

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

@app.callback([Output('button', 'style'),Output('button', 'n_clicks')], [Input('button', 'n_clicks')])
def change_button_style(n_clicks):

    if n_clicks > 0:
        n_clicks = 1
        return blue_style,n_clicks

    else:
        return green_style, n_clicks

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

@app.callback([Output('PCA', 'figure'),Output('TSNE', 'figure')],
              Input('datatable-advanced-filtering', 'derived_virtual_data')
)
def update_graph(derived_virtual_data):
    features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8', 'T cells CD4 naive', 'T cells CD4 memory resting', 
    'T cells CD4 memory activated', 'T cells follicular helper', 'T cells regulatory (Tregs)', 'T cells gamma delta', 
    'NK cells resting', 'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1', 'Macrophages M2', 
    'Dendritic cells resting', 'Dendritic cells activated', 'Mast cells resting', 'Mast cells activated', 'Eosinophils', 
    'Neutrophils', 'Cell_Cycle', 'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53', 'WNT', 'Hypoxia', 
    'SRC', 'ESR1', 'ERBB2', 'PROLIF']
    data = derived_virtual_data
    target = ['PAM50']
    # Separating out the features
    data = pd.DataFrame(data)
        # Separating out the features
    x = data[features].values

    # Separating out the target
    y = data.loc[:,target]

    return reduce_dimension('pca', dimension=2, x=x, y=y, kernel=None), reduce_dimension('tsne', dimension=2, x=x, y=y, kernel=None)
    
    
def description_welcome():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Br(),
            html.H2("Welcome to the Clinical Analytics Dashboard"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
        ],
    )


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

@app.callback(
	Output('pathways_heatmap_output', 'figure'),
	[Input('select_pathways', 'value')])
def update_figure(value):
    if value is None:
        return {'data': []}
    else:  
        return heatmap(value)

@app.callback(
	Output('cells_heatmap_output', 'figure'),
	[Input('select_cells', 'value')])
def update_figure(value):
    if value is None:
        return {'data': []}
    else:  
        return heatmap(value)

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

@app.callback(Output('SHAP', 'figure'),
	[Input('radio', 'value'),Input('button', 'n_clicks')])
def update_figure(value,n_clicks):
    if n_clicks > 0:
        return SHAP_VALUES(value)
    else:  
        return {'data': []}

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
        df = pd.read_csv(DATA_PATH)

        kmf = lifelines.KaplanMeierFitter()
        TBasal = df[df.PAM50 == button2]["RFS"]
        EBasal = df[df.PAM50 == button2]["RFSE"]  
        kmf.fit(TBasal, event_observed=EBasal)
        
        return "Result: " + str(round(kmf.predict(int(input1))*100,5)) + "%"

@app.callback(
    Output("significance", "children"),
    Input("drop11", "value"),Input("drop21", "value"),
)
def u(drop11,drop21):
    df = pd.read_csv(DATA_PATH)
    T1 = df[df.PAM50 == drop11]["RFS"]
    E1 = df[df.PAM50 == drop11]["RFSE"]
    T2 = df[df.PAM50 == drop21]["RFS"]
    E2 = df[df.PAM50 == drop21]["RFSE"]
    results=logrank_test(T1,T2, event_observed_A=E1, event_observed_B=E2)
    
    if results.p_value <= 0.05:
        text = "Significant difference      " + "p-value < 0.05: " + str(round(results.p_value,2))
    else:
        text = "No significant difference      " + "p-value > 0.05: " + str(round(results.p_value,2))
    return text

@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):

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

    if tab == 'heatmap':
        return  html.Div([
                    html.Div([    
                        html.Div([
                            html.Br()
                        ]),
                    ], className="row"),
                    html.Div([    
                        html.Div([                             
                        ], className="six columns" , style={'width':'5%','text-align':'center'}),
                        html.Div([   
                            html.Br(),html.Br(),           
                            html.H2("Pathways"),
                            html.P("Select a set of pathways"),
                            dcc.Dropdown(
                                id="select_pathways",
                                options=[{"label": i, "value": i} for i in pathways],
                                value=pathways,
                                multi=True,
                            ),
                            html.Br(), html.Br(), html.Br(),
                        ], className="six columns" , style={'width':'20%'}),
                        html.Div([                             
                        ], className="six columns" , style={'width':'5%','text-align':'center'}),
                        html.Div([  
                            html.Br(),html.Br(),html.Br(),           
                            dcc.Graph(id='pathways_heatmap_output')
                        ], className="six columns" , style={'width':'70%','text-align':'center'}),
                        html.Div([    
                        ], className="six columns" , style={'width':'5%','text-align':'center'})
                    ], className="row", style=dict(display='flex')),
                    html.Div([    
                        html.Div([                             
                        ], className="six columns" , style={'width':'5%','text-align':'center'}),
                        html.Div([  
                            html.Br(),html.Br(),             
                            html.H2("Immune cells"),
                            html.P("Select a set of immune cells"),
                            dcc.Dropdown(
                                id="select_cells",
                                options=[{"label": i, "value": i} for i in cells],
                                value=cells,
                                multi=True,
                            ),
                            html.Br(), html.Br(), html.Br(),
                            
                        ], className="six columns" , style={'width':'20%'}),
                        html.Div([                             
                        ], className="six columns" , style={'width':'5%','text-align':'center'}),
                        html.Div([  
                            html.Br(),html.Br(),html.Br(),           
                            dcc.Graph(id='cells_heatmap_output')
                        ], className="six columns" , style={'width':'70%','text-align':'center'}),
                        html.Div([    
                        ], className="six columns" , style={'width':'5%','text-align':'center'})
                    ], className="row", style=dict(display='flex'))
                ])
    elif tab == 'hello':
        return welcome_tab()
    elif tab == 'dim':
        return dim_tab()
    elif tab == 'corr':
        #return  html.Div([
                    #html.Div([    
                        #html.Div([
                            #html.Br(),      
                            #html.H1("Hierarchical Clustering",style={'text-align':'center'})
                        #]),
                    #], className="row"),
                    #html.Div([    
                        #html.Div([                             
                        #], className="six columns" , style={'padding':10,'width':'5%','text-align':'center'}),
                        #html.Div([              
                            #html.H2("Pathways"),
                            #html.P("Select Admit Source"),
                            #dcc.Dropdown(
                                #id="isv_select",
                                #options=[{"label": i, "value": i} for i in pathways],
                                #value=pathways,
                                #multi=True,
                            #),
                            #html.Br(), 
                        #], className="six columns" , style={'padding':10,'width':'45%','text-align':'center'}),
                        #html.Div([
                                  
                            #html.H2("Immune Cells"),
                            #html.P("Select Admit Source"),
                            #dcc.Dropdown(
                            #    id="admit-selectbsdb",
                            #    options=[{"label": i, "value": i} for i in cells],
                            #    value=cells,
                            #    multi=True,
                            #),
                            #html.Br()
                        #], className="six columns", style={'padding':10,'width':'45%','text-align':'center'}),
                        #html.Div([    
                        #], className="six columns" , style={'padding':10,'width':'5%','text-align':'center'})
                    #], className="row", style=dict(display='flex'))
                #])
        return html.H1("Under Construction 🚧")
    elif tab == 'model1':
        return shap_tab()
    
    elif tab == 'model2':
        return kaplan_tab()

if __name__ == '__main__':
    app.run_server(debug=True)
