



import plotly
import plotly.graph_objs as go

import numpy as np
import pandas as pd


from sklearn.manifold import TSNE


def make_TSNE_plot(features, target, return_fig=True):

    if isinstance(features, pd.DataFrame):
        features_np = features.to_numpy()

    elif isinstance(features, np.ndarray):
        features_np = features

    if isinstance(target, pd.Series):
        target_df = target.to_frame()
    elif isinstance(target, pd.DataFrame):
        target_df = target
    elif isinstance(target, np.ndarray):
        target_df = pd.DataFrame(data=target, columns = ["target"])

    tsne_features = TSNE(n_components=2, random_state=42).fit_transform(features_np)

    tsne_features_df = pd.DataFrame(data=tsne_features, columns = ["tsne_1", "tsne_2"])

    output_df = pd.concat([tsne_features_df, target_df], axis=1)

    color = output_df.iloc[:,2]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=output_df.loc[:,"tsne_1"],
            y=output_df.loc[:,"tsne_2"],
            mode="markers",
            marker=dict(color=color, colorscale="Viridis", showscale=True)
            )
        )

    try:
        title = "TSNE plot"
        fig.update_layout(title=title)
    except ValueError:
        pass
    except NameError:
        pass
    except AttributeError:
        pass

    if return_fig:
        return fig
    else:
        plotly.io.write_html(fig, file="tsne_plot.html", auto_open=True)
        return None




data = pd.read_parquet("data/ChemicalManufacturingProcess.parquet")


target = data.loc[:,"Yield"]
features = data.drop("Yield", axis=1)


make_TSNE_plot(features=features, target=target, return_fig=False)











