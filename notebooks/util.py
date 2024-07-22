from pathlib import Path

import joblib
import pandas as pd


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
