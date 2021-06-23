import dash
import base64
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import aux

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


def welcome_tab():
    return html.Div([
                        html.Div([html.Br(),html.Br(),
                html.Img(src=b64_image("img/portada.png"), style={'height':'70%'})], style={'width':'40%','text-align':'center'})
                ,html.Div([
                html.Br(),html.Br(),
                html.H3("📜 Abstract"),
                html.Div(dcc.Markdown('''

                ---

                *This study is focused on the detection of relevant modulators from
                genetic expressions in order to help in the analysis and treatment of diagnosed breast
                cancer patients. The identification of the genes will be established from those that con-
                tribute in the classification of breast tumor subtypes and the ones that take part in the
                prediction of a possible relapse. Two methodologies will be used to find the results: 
                Firstly, the neural networks will be used in the classification model. Then, we will apply inter-
                pretability techniques in order to provide validity to the result, allowing us to extract the
                pathways that have largely determined the output. Secondly, we have worked with the
                previous idea to compute the relapse prediction, but the results were not as good as we
                expected. However, two estimators that are able to model the patients relapse have been
                constructed. The contribution of each variable will be studied in order to establish which
                modulators have more prominence when modeling the risk of relapse.*

                ---

                This project has been developed in the context of a collaboration agreement
                between IDIBELL institution and the research group SPCOM (Signal Processing and
                Communications) from the UPC.
                ''')),
                html.Div([html.Br(),
                html.Img(src=b64_image("img/logos.png"), style={'height':'70%', 'width':'70%','text-align':'center'})])], style={'width':'50%'})],style={'display':'flex'})
    
                




        
        