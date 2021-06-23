import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

import aux

def description_data():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Br(),
            html.H2("Explore the modulators in Breast Cancer Tumors"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
        ],
    )

def welcome_tab():
    return html.Div([
            html.Div(
            id="app-container",
            children=[
                # Left column
                html.Div(
                    id="left-column",
                    className="four columns",
                    children=[description_data()]
                    + [
                        html.Div(
                            ["initial child"], id="output-clientside", style={"display": "none"}
                        )
                    ],
                ),
            ],
            )
        ])