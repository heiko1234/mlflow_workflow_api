

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np




def validation_plot(df_original, df_predicted):

    try:

        fig_output = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.025, shared_xaxes=True)


        # Main plot
        fig_output.add_trace(go.Scatter(x=df_original.index, y=df_original, mode='markers', marker=dict(color='blue', size = 12), name='original'), row=1, col=1)
        fig_output.add_trace(go.Scatter(x=df_predicted.index, y=df_predicted, mode='markers', marker=dict(color='red', size = 12), name='prediction'), row=1, col=1)

        # Residuals
        diff = df_original - df_predicted
        fig_output.add_trace(go.Scatter(x=diff.index, y=diff, mode='markers', marker=dict(color='black', size = 12), name='diff'), row=2, col=1)
        fig_output.update_layout(title='Original vs. Predicted Values', xaxis_title='Index', yaxis_title=df_original.name)

    except Exception as e:
        print(e)
        fig_output = None

    return fig_output





def x_y_plot(df_original, df_predicted):

    try:

        fig_output = go.Figure()

        fig_output.add_trace(go.Scatter(x=df_original, y=df_predicted, mode='markers', marker=dict(color='black', size = 12), name='original vs. predicted'))

        # add line with 45 degree
        fig_output.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])

        fig_output.update_layout(title='Original vs. Predicted Values', xaxis_title='Original', yaxis_title='Predicted')

    except Exception as e:
        print(e)
        fig_output = None

    return fig_output



