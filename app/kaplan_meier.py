import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly
import lifelines
from aux import features, params

import numpy as np
import pandas as pd

DATA_PATH = "data/clean_train.csv"

def kaplan_meier(df):
    fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("", "Patients at risk"), vertical_spacing=0.39)
    kmfs = []

    steps = 5 # the number of time points where number of patients at risk which should be shown

    x_min = 0 # min value in x-axis, used to make sure that both plots have the same range
    x_max = 0 # max value in x-axis

    fig.update_layout(template='ggplot2', height=800, width=1200,paper_bgcolor="rgb(247,247,247)")


    lcolors = ['#636EFA','#FF6692','#00CC96','#AB63FA']
    bcolors = ['rgba(158,185,243, 0.4)','rgba(253,218,236,0.4)','rgba(179,226,205, 0.4)','rgba(222,203,228, 0.4)']
    i = 0                 
    for PAM50 in df.PAM50.unique():
        kmf = lifelines.KaplanMeierFitter()
        if PAM50 == 'Basal':
            TBasal = df[df.PAM50 == PAM50]["RFS"]
            EBasal = df[df.PAM50 == PAM50]["RFSE"]  
            kmf.fit(TBasal, event_observed=EBasal)

        if PAM50 == 'Her2':
            THer2 = df[df.PAM50 == PAM50]["RFS"]
            EHer2 = df[df.PAM50 == PAM50]["RFSE"]
            kmf.fit(THer2, event_observed=EHer2)

        if PAM50 == 'LumA':
            TLumA = df[df.PAM50 == PAM50]["RFS"]
            ELumA = df[df.PAM50 == PAM50]["RFSE"]
            kmf.fit(TLumA, event_observed=ELumA)

        if PAM50 == 'LumB':
            TLumB = df[df.PAM50 == PAM50]["RFS"]
            ELumB = df[df.PAM50 == PAM50]["RFSE"]
            kmf.fit(TLumB, event_observed=ELumB)

        kmfs.append(kmf)
        x_max = max(x_max, max(kmf.event_table.index))
        x_min = min(x_min, min(kmf.event_table.index))

        
        fig.add_trace(plotly.graph_objs.Scatter(x=kmf.survival_function_.index,
                                                    y=kmf.confidence_interval_.values[:,0], line=dict(width=0), showlegend=False,
                                                    name=PAM50), 
                        1, 1)
        fig.add_trace(plotly.graph_objs.Scatter(x=kmf.survival_function_.index,
                                                    y=kmf.confidence_interval_.values[:,1], line=dict(width=0), showlegend=False,
                                                    name=PAM50,fillcolor=bcolors[i],
        fill='tonexty',), 
                        1, 1)
        fig.append_trace(plotly.graph_objs.Scatter(x=kmf.survival_function_.index,
                                                    y=kmf.survival_function_.values.flatten(), line=dict(color=lcolors[i]),
                                                    name=PAM50), 
                        1, 1)

        i += 1
        
    for s, PAM50 in enumerate(df.PAM50.unique()):
        x = []
        kmf_ = kmfs[s].event_table
        for i in range(0, int(x_max), int(x_max / (steps - 1))):
            x.append(kmf_.iloc[np.abs(kmf_.index - i).argsort()[0]].name)
        fig.append_trace(plotly.graph_objs.Scatter(x=x, 
                                                y=[PAM50 + " "] * len(x), 
                                                text=[kmfs[s].event_table[kmfs[s].event_table.index == t].at_risk.values[0] for t in x], 
                                                mode='text', 
                                                showlegend=False), 
                        2, 1)

    fig.update_yaxes(title_text="Tumor", row=2, col=1)
    fig.update_yaxes(title_text="Relapse Probability", row=1, col=1)
    fig.update_xaxes(title_text="Time (RFS)", row=2, col=1) 

    # just a dummy line used as a spacer/header
    t = [''] * len(x)
    fig.append_trace(plotly.graph_objs.Scatter(x=x, 
                                            y=[''] * (len(x)-20), 
                                            text=t,
                                            mode='text', 
                                            showlegend=False), 
                    2, 1)

        
    # prettier layout
    x_axis_range = [x_min - x_max * 0.05, x_max * 1.05]
    fig['layout']['xaxis2']['visible'] = True
    fig['layout']['xaxis2']['range'] = x_axis_range
    fig['layout']['xaxis']['range'] = x_axis_range
    fig['layout']['yaxis']['domain'] = [0.4, 1]
    fig['layout']['yaxis2']['domain'] = [0.0, 0.3]
    fig['layout']['yaxis2']['showgrid'] = False
    fig['layout']['yaxis']['showgrid'] = True

    return fig

def kaplan_tab():
    df = pd.read_csv(DATA_PATH)

    return  html.Div([html.Br(),
                        html.Div([
                            html.Div([], style={'width':'8%','text-align':'center'}),
                            html.Div(html.Div([html.Div([html.H2("Kaplan-Meier Curve"),dcc.Graph(figure=kaplan_meier(df)),html.Div([],style=dict(display='flex'))]),html.Div([html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.P("Relapse Probability prediction:"),dcc.Input(id="input1", type="text", placeholder="Time"),html.Br(),html.Br(),dcc.RadioItems(
                                                    id='button2',
                                                    options=[
                                                        {'label': 'Basal', 'value': 'Basal'},
                                                        {'label': 'Her2', 'value': 'Her2'},
                                                        {'label': 'LumA', 'value': 'LumA'},
                                                        {'label': 'LumB', 'value': 'LumB'}
                                                    ],
                                                    value='Basal',
                                                    labelStyle={'display': 'block'}),html.Br(),html.Br(),html.Div(id="outputtext"),html.Br(),html.Br(),html.Br(),html.P("Log-Rank Test:"), dcc.Dropdown(
    id='drop11',
    options=[
        {'label': 'Basal', 'value': 'Basal'},
        {'label': 'Her2', 'value': 'Her2'},
        {'label': 'LumA', 'value': 'LumA'},
        {'label': 'LumB', 'value': 'LumB'},
    ],
    value='Basal',
    clearable=False
)  ,html.Div([], style={'width':'10%','text-align':'center'}),dcc.Dropdown(
    id='drop21',
    options=[
        {'label': 'Basal', 'value': 'Basal'},
        {'label': 'Her2', 'value': 'Her2'},
        {'label': 'LumA', 'value': 'LumA'},
        {'label': 'LumB', 'value': 'LumB'},
    ],
    value='Her2',
    clearable=False
),html.Br(),html.Div(id="significance")])], style=dict(display='flex')), style={'width':'70%'}),
                            html.Div([], style={'width':'10%','text-align':'center'})], style=dict(display='flex')),
                    ])