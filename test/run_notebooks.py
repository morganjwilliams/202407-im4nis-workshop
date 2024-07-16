from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

# iterate through the notebooks, and catch cell exectuion errors to look at later
exceptions = {}
for nbpath in Path("./notebooks").glob("*.ipynb"):
    print("Running {}".format(nbpath))
    with open(nbpath) as f:
        nb = nbformat.read(f, as_version=4)
    try:
        ep.preprocess(nb, {"metadata": {"path": "./notebooks/"}})
    except CellExecutionError as e:
        print("Error in {}".format(nbpath))
        exceptions[nbpath] = e

for k, e in exceptions.items():
    print(k)
    print(e)


assert not len(exceptions), "Some notebooks errored: {}".format(
    ", ".join([p.name for p in exceptions.keys()])
)
