import logging
from pathlib import Path
import requests
import shutil
from typing import Any, Callable, Dict, Optional

from pathos import logger
from pathos.core import getpid
from retry import retry

@retry(tries=10, delay=5, backoff=10)
def download_image(
    i_ee,
    index: int,
    image_download_method: Callable,
    serialized_image_list: Dict[str, Any],
    name_prefix: str,
    out_dir: Optional[Path],
    download_kwargs: Optional[Dict[str, Any]]
) -> None:
    """
    Hidden function to be used with download_image_collection. As we want compatibility with
    windows, we need to use dill to pickle the initialized earthengine module. Pathos uses
    dill pickle data uses to start new processes with its multiprocessing pool, while the python
    multiprocessing module uses pickle, which cannot pickle module objects.
    """

    if not out_dir:
        out_dir: Path = Path.cwd() / "output"
    if not download_kwargs:
        download_kwargs: Dict[str, str] = {}

    pid = getpid()
    log_file: Path = out_dir / "logging" / f"{pid}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)

    plogger = logger(level=logging.DEBUG, handler=fh)

    ee_api_url: str = i_ee.data._cloud_api_base_url
    ee_hv_url: str = "https://earthengine-highvolume.googleapis.com"
    if ee_api_url is not ee_hv_url:
        i_ee.Initialize(opt_url=ee_hv_url)
    
    image_list: i_ee.List = i_ee.deserializer.fromJSON(serialized_image_list)

    img: i_ee.Image = i_ee.Image(image_list.get(index))
    url: str = image_download_method(img, download_kwargs)
    r: requests.Response = requests.get(url, stream=True)

    # File format chain
    format: Optional[str] = download_kwargs.get("format")
    if format:
        if format == "GEO_TIFF":
            extention: str = ".tif"
        elif format == "NPY":
            extention: str = ".npy"
        elif format == "PNG":
            extention: str = ".png"
        elif format == "JPG":
            extention: str = ".jpg"
        else:
            extention: str = ".tif.zip"
    elif image_download_method == i_ee.Image.getDownloadURL:
        extention: str = ".tif.zip"
    elif image_download_method == i_ee.Image.getThumbURL:
        extention: str = ".png"
    else:
        raise RuntimeError(
            f"image download method {image_download_method} unknown.")

    # Image naming chain
    img_props: Dict[str, Any] = img.getInfo()['properties']
    t0: Optional[int] = img_props.get("system:time_start")
    img_index: Optional[int] = img_props.get("system:index")
    if t0:
        file_id: str = t0
    elif img_index:
        file_id: str = img_index
    else:
        file_id: str = index

    # File name
    filename: Path = out_dir / f"{name_prefix}{file_id}{extention}"

    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    
    plogger.info(f"Done: {index}")
    plogger.removeHandler(fh)
