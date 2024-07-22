from pathlib import Path
from pydoc import locate

import nbformat
import yaml
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

if __file__ is not None:
    dirpath = Path(__file__).parent
else:
    dirpath = Path("./")
with open(dirpath / "config.yaml", "r") as f:
    test_config = yaml.safe_load(f)
    print(test_config)

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
        print("Error in {}: {}".format(nbpath, str(e.ename)))
        exceptions[nbpath] = e.ename

unexpected_erorrs = {}
for k, e in exceptions.items():
    if k.stem not in test_config.get("allow-fail", {}).keys():
        unexpected_erorrs[k] = e
    else:
        _allowed_errors = test_config.get("allow-fail").get(k.stem)
        if str(e) not in _allowed_errors:
            unexpected_erorrs[k] = e


assert not len(unexpected_erorrs), "Some notebooks errored: {}".format(
    ", ".join([p.name for p in unexpected_erorrs.keys()])
)
