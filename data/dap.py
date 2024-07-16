import os
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from tqdm.auto import tqdm


def download_from_DAP_links(
    filepath,
    output_dir=None,
    collapse_data=True,
    collapse_metadata=False,
    fltr=None,
):
    """
    Download a selection of files from DAP,
    based on the provided list of file links.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to the file manifest.
    output_dir : str | pathlib.Path
        Output directory, if not the current directory.
    collapse_data : bool
        Whether to collapse the 'data' folder in DAP archives,
        and place this at the root.
    collapose_metadata : bool
        Whether to collapse the 'metadata' folder in DAP archives,
        and place this at the root.
    fltr : str
        Pattern to match keys on, to only download a subset.

    Returns
    -------
    dict
        Items downloaded with associated metadata and local filepaths.
    """
    with open(filepath, "r") as f:
        data = f.read()

    if output_dir is None:
        output_dir = "./"
    output_dir = Path(output_dir)

    items = {}
    for line in data.splitlines():
        if line.startswith("#"):
            pass
        elif line.startswith("https"):
            url = line
        else:
            dirpath = line.strip().replace("dir=", "")
            _url = urlparse(url)
            parts = Path(_url.path).parts
            # first three parts of tail are "/", "dapprd" and "<ID>v<Version>"
            _project_id = parts[2]
            keyparts = parts[3:]
            if (collapse_data and keyparts[0] == "data") or (
                collapse_metadata and keyparts[0] == "metadata"
            ):
                keyparts = keyparts[1:]

            dirparts = Path(dirpath).parts[1:]
            if not dirparts:
                dirparts = [""]
            if (collapse_data and dirparts[0] == "data") or (
                collapse_metadata and dirparts[0] == "metadata"
            ):
                dirparts = dirparts[1:]
            items[os.path.join(*keyparts).replace(" ", "").replace("%20", "")] = {
                "url": url,
                "name": parts[-1].replace(" ", "").replace("%20", ""),
                "dir": os.path.join(*dirparts).replace(" ", "").replace("%20", "")
                if dirparts
                else None,
            }
    if fltr is not None:
        items = {k: d for k, d in items.items() if re.search(fltr, k)}

    pbar = tqdm(items.items())
    for k, d in pbar:
        tgt = output_dir / Path(d["dir"]) / d["name"]
        pbar.set_description("Downloading: {}".format(tgt.name))

        tgt.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(d["url"]) as stream:
            data = stream.read()

        with open(tgt, "wb") as f:
            f.write(data)

        d["path"] = tgt

    return items
