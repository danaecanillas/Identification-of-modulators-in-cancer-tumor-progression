import dash_html_components as html
import dash_core_components as dcc
import aux
from dash.dependencies import Input, Output

def hierarchical_tab():
    return html.Div([
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
                            options=[{"label": i, "value": i} for i in aux.pathways],
                            value=aux.pathways,
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
                            options=[{"label": i, "value": i} for i in aux.cells],
                            value=aux.cells,
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