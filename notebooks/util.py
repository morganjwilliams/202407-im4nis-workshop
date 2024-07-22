from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrolite.plot.color import process_color
from pyrolite.util.plot.axes import init_axes
from pyrolite.util.plot.legend import proxy_line
from pyrolite.util.plot.style import mappable_from_values
from sklearn.inspection import permutation_importance

AUS_STATES = gpd.read_file("../data/STE_2021_AUST_SHP_GDA2020.zip")


def get_model_data(name, dirpath="../data/MachineLearningModels"):
    """
    Convenience function to do a quick lookup of data files by name,
    and load the relevant items.
    """
    dirpath = Path(dirpath)
    return {
        p.stem.replace(name + "_", ""): (
            pd.read_csv(p) if p.suffix == ".csv" else joblib.load(p)
        )
        for p in dirpath.rglob("{}*".format(name))
    }


def map_markersizes(arr, minsize=5, maxsize=100, abs=False):
    """
    Map an array of markersizes to an array of values.
    """
    if abs:
        _arr = np.abs(arr)
    else:
        _arr = arr
    return minsize + (_arr - _arr.min()) / (_arr.max() - _arr.min()) * (
        maxsize - minsize
    )


def markersize_mapper(arr, abs=False, minsize=5, maxsize=100):
    if abs:
        _arr = np.abs(arr)
    else:
        _arr = arr

    def map_markersizes(vals):
        """
        Map an array of markersizes to an array of values.
        """
        return minsize + (vals - _arr.min()) / (_arr.max() - _arr.min()) * (
            maxsize - minsize
        )

    return map_markersizes


def plot_sample_predictions(
    predictions_by_sample,
):
    fig, ax = plt.subplots(1, figsize=(8, 8))

    AUS_STATES[AUS_STATES.intersects(predictions_by_sample.unary_union)].plot(
        color="0.9", ax=ax, edgecolor="0.8"
    )
    predictions_by_sample.plot(
        ax=ax,
        c=process_color(c=predictions_by_sample["prop_min"], cmap="cividis", alpha=0.8)[
            "c"
        ],
        markersize=map_markersizes(
            np.log(predictions_by_sample["counts"]), maxsize=200
        ),
        marker="o",
    )

    ax.grid(alpha=0.25)

    ax.set(aspect="equal", ylabel="Latitude", xlabel="Longitude")
    plt.colorbar(
        mappable_from_values(predictions_by_sample["prop_min"], cmap="cividis"),
        ax=ax,
        label="Proportion of Mineralized Grains",
    )

    count_legend = np.array([10, 100, 1000, 5000])
    hdls = [
        proxy_line(marker="o", markersize=np.sqrt(ms), linewidth=0, color="k")
        for ms in markersize_mapper(np.log(predictions_by_sample["counts"]))(
            np.log(count_legend)
        )
    ]
    _leg = plt.legend(hdls, count_legend, bbox_to_anchor=(0, 1), title="Spinel Grains")
    return fig, ax
