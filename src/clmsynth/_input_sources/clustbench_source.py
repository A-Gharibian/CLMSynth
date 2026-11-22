# SPDX-License-Identifier: MIT
# clustbench_source.py
"""The clustbench source: Gagolewski's clustering benchmark datasets, fetched over
HTTP(S), with their batteries and descriptions. ``dataset_sources`` is its interface
and re-exports these names.
The data repository is AGPL-3.0, and its datasets are provided for research purposes only unless their
  description says otherwise: This module interfaces with the repository.
"""

import gzip
import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final

import numpy as np
import pandas as pd

from . import COHORT_COLUMN, GROUND_TRUTH_PREFIX, Source

log: logging.Logger = logging.getLogger(__name__)

# Seconds one network operation of a download may wait: connecting, or one read.
CLUSTBENCH_TIMEOUT = 30
# Seconds a whole file may take: a host that sends a few bytes within every
# CLUSTBENCH_TIMEOUT would otherwise hold a run without end.
CLUSTBENCH_DEADLINE = 600
# About 3.5 times the largest v1.1.0 file, mnist/fashion.data.gz (36.7 MiB, about
# 148 MiB unpacked).
CLUSTBENCH_MAX_DOWNLOAD = 128 * 2**20
CLUSTBENCH_MAX_UNPACKED = 512 * 2**20

CLUSTBENCH_METADATA: Final[Mapping[str, Mapping[str, str]]] = {
    "wut": {
        "description": "Warsaw University of Technology. Covers spatial, density-based, and overlapping clusters.",
        "examples": "smile, x1, x2, twonorm"
    },
    "sipu": {
        "description": "SIPU datasets (Fränti & Sieranoja)."
                       "Features varying densities, high dimensions, and large cluster numbers.",
        "examples": "a1, s1, unbalance, birch1"
    },
    "fcps": {
        "description": "Fundamental Clustering Problem Suite (A. Ultsch)."
                       "Classic 2D/3D datasets with distinct geometric challenges.",
        "examples": "atom, chainlink, hepta, lsun"
    },
    "graves": {
        "description": "Features complex shapes, heavy noise, and non-linear boundaries (like zigzags).",
        "examples": "zigzag_noisy, ring_noisy, dense_disk"
    },
    "other": {
        "description": "Assorted community datasets, including those specifically designed to test HDBSCAN.",
        "examples": "hdbscan, chameleon_t8_8k"
    },
    "uci": {
        "description": "UCI Machine Learning Repository. Eight real classification datasets in 7 to 60 "
                       "dimensions, whose classes clustering often finds hard.",
        "examples": "ecoli, wine, yeast"
    },
    "mnist": {
        "description": "MNIST handwritten digits and Fashion-MNIST articles: 28x28 greyscale images "
                       "as 784 features, 10 classes.",
        "examples": "digits, fashion"
    },
    "g2mg": {
        "description": "Two equal-sized Gaussian clusters, 2048 points in 1 to 128 dimensions, named "
                       "g2mg_<dim>_<scale>; the _1_ ones have a single feature.",
        "examples": "g2mg_2_10, g2mg_4_20"
    },
    "h2mg": {
        "description": "Two equal-sized Gaussian-like hubs, each point on a sphere around its "
                       "cluster's center, 2048 points in 1 to 128 dimensions, named "
                       "h2mg_<dim>_<scale>; the _1_ ones have a single feature.",
        "examples": "h2mg_2_10, h2mg_4_20"
    },
}

CLUSTBENCH_DATASETS: Final[Mapping[str, Sequence[str]]] = {
    "wut": [
        "circles", "cross", "graph", "isolation", "labirynth", "mk1", "mk2",
        "mk3", "mk4", "olympic", "smile", "stripes", "trajectories",
        "trapped_lovers", "twosplashes", "windows", "x1", "x2", "x3",
        "z1", "z2", "z3",
    ],
    "sipu": [
        "a1", "a2", "a3", "aggregation", "birch1", "birch2", "compound",
        "d31", "flame", "jain", "pathbased", "r15", "s1", "s2", "s3", "s4",
        "spiral", "unbalance", "worms_2", "worms_64",
    ],
    "fcps": [
        "atom", "chainlink", "engytime", "hepta", "lsun", "target",
        "tetra", "twodiamonds", "wingnut",
    ],
    "graves": [
        "dense", "fuzzyx", "line", "parabolic", "ring", "ring_noisy",
        "ring_outliers", "zigzag", "zigzag_noisy", "zigzag_outliers",
    ],
    "other": [
        "chameleon_t4_8k", "chameleon_t5_8k", "chameleon_t7_10k",
        "chameleon_t8_8k", "hdbscan", "iris", "iris5", "square",
    ],
    "uci": [
        "ecoli", "glass", "ionosphere", "sonar", "statlog", "wdbc",
        "wine", "yeast",
    ],
    "mnist": ["digits", "fashion"],
    "g2mg": [f"g2mg_{d}_{s}" for d in (1, 2, 4, 8, 16, 32, 64, 128)
             for s in range(10, 91, 10)],
    "h2mg": [f"h2mg_{d}_{s}" for d in (1, 2, 4, 8, 16, 32, 64, 128)
             for s in range(10, 91, 10)],
}

CLUSTBENCH_HEAVY: Final[AbstractSet[str]] = {"mnist", "g2mg", "h2mg"}


class _HttpRedirectsOnly(urllib.request.HTTPRedirectHandler):
    """Follows a redirect to http(s) only; urllib's own handler also follows one to ftp."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        scheme = urllib.parse.urlparse(newurl).scheme
        if scheme not in ("http", "https"):
            raise urllib.error.HTTPError(
                newurl, code, f"{msg} - redirect to {scheme!r} refused: only http(s) is fetched",
                headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpRedirectsOnly)


def _read_whole(resp, url: str) -> bytes:
    """The body of `resp`, within CLUSTBENCH_DEADLINE and CLUSTBENCH_MAX_DOWNLOAD."""
    deadline = time.monotonic() + CLUSTBENCH_DEADLINE
    # read1 returns what one read brings; read(n) would wait for all n bytes.
    read = getattr(resp, "read1", resp.read)
    chunks: list[bytes] = []
    size = 0
    while chunk := read(1 << 16):
        size += len(chunk)
        if size > CLUSTBENCH_MAX_DOWNLOAD:
            raise ValueError(f"clustbench: {url} is larger than {CLUSTBENCH_MAX_DOWNLOAD} bytes; "
                             "not read further.")
        if time.monotonic() > deadline:
            raise TimeoutError(f"clustbench: {url} was not downloaded within "
                               f"{CLUSTBENCH_DEADLINE} s.")
        chunks.append(chunk)
    return b"".join(chunks)


def _timed_out(error: BaseException) -> bool:
    """A time-out, raised as it is or as the reason of urllib's URLError."""
    return isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None),
                                                         TimeoutError)


def _loadtxt_url(url: str, **loadtxt_kwargs) -> np.ndarray:
    """
    Fetches a gzipped text array over HTTP(S), then parses it. Each network
    operation waits at most CLUSTBENCH_TIMEOUT and the whole file
    CLUSTBENCH_DEADLINE; the file and what it unpacks to are capped by
    CLUSTBENCH_MAX_DOWNLOAD and CLUSTBENCH_MAX_UNPACKED.

    `base_url` reaches here from the config, and a config is a shareable
    artifact (see SECURITY.md), so a `file://` or `data:` URL would turn a
    dataset fetch into a local read. Only http(s) is opened, redirects included.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(
            f"clustbench: refusing to fetch {url!r}: only http(s) is fetched, "
            f"not {scheme!r}. Check clustbench_suite.base_url.")
    try:
        response = _OPENER.open(url, timeout=CLUSTBENCH_TIMEOUT)  # nosec B310
    except urllib.error.HTTPError as error:
        # An HTTPError owns a response too, but is raised before the with below.
        error.close()
        raise
    with response as resp:
        raw = _read_whole(resp, url)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        text = stream.read(CLUSTBENCH_MAX_UNPACKED + 1)
    if len(text) > CLUSTBENCH_MAX_UNPACKED:
        raise ValueError(f"clustbench: {url} unpacks to more than {CLUSTBENCH_MAX_UNPACKED} "
                         "bytes; not read.")
    return np.loadtxt(io.BytesIO(text), **loadtxt_kwargs)


def fetch_clustbench_labelings(
        dataset_group: str,
        dataset_name: str,
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0",
        max_labelings: int = 10,
) -> dict[str, np.ndarray]:
    """Downloads every reference labeling (labels0, labels1, ...) of one
    clustbench dataset, up to the first index the server does not have (404).
    A labeling that fails otherwise is skipped and named in one warning; at a
    time-out no later labeling is asked for."""
    labeling, failed, stopped = _labelings(dataset_group, dataset_name, base_url, max_labelings)
    if failed:
        log.warning(f"{dataset_group}/{dataset_name}: could not fetch {', '.join(failed)}; "
                    f"continuing with {len(labeling)} labeling(s)." + _stopped_note(stopped))
    return labeling


def _stopped_note(stopped: bool) -> str:
    return " Stopped at the time-out: no later labeling was asked for." if stopped else ""


def _labelings(dataset_group: str, dataset_name: str, base_url: str,
               max_labelings: int = 10) -> tuple[dict[str, np.ndarray], list[str], bool]:
    """The labelings fetch_clustbench_labelings returns, those that failed to download
    (other than at the first 404), and whether it stopped at a time-out. Logs only a
    missing labels0."""
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"
    labeling: dict[str, np.ndarray] = {}
    failed: list[str] = []
    stopped = False
    for i in range(max_labelings):
        label_name = f"labels{i}"
        try:
            # ndmin: a one-point dataset's labeling is a list of one, not a bare number.
            labeling[label_name] = _loadtxt_url(f"{target_path}.{label_name}.gz", dtype="int",
                                                ndmin=1)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                failed.append(f"{label_name} ({e})")
                continue
            if i == 0:
                log.error(f"No reference labels found for {dataset_group}/{dataset_name}.")
            break
        except Exception as e:
            failed.append(f"{label_name} ({e})")
            if _timed_out(e):
                # A host that stopped answering would keep each later labeling waiting as long.
                stopped = True
                break
    return labeling, failed, stopped


def fetch_clustbench_data(
        dataset_group: str = "wut",
        dataset_name: str = "smile",
        base_url: str = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"
) -> pd.DataFrame | None:
    """Fetches one Gagolewski benchmark dataset (features + all usable
    labeling) into the standard fetcher frame; None on failure, a labeling that
    fails to download included."""
    if dataset_group not in CLUSTBENCH_METADATA:
        log.warning(f"Battery '{dataset_group}' is not in the recommended list, but attempting to fetch anyway.")

    log.info(f"Fetching external clustering dataset: {dataset_group}/{dataset_name}...")
    target_path = f"{base_url}/{dataset_group}/{dataset_name}"

    try:
        data = _loadtxt_url(f"{target_path}.data.gz", ndmin=2)
    except Exception as e:
        log.error(f"Failed to fetch dataset features: {e}")
        return None

    labeling, failed, stopped = _labelings(dataset_group, dataset_name, base_url)
    if failed:
        # Written without it, a later labeling would take its Cluster_n name.
        log.error(f"Failed to fetch {dataset_group}/{dataset_name}: could not fetch "
                  f"{', '.join(failed)}.{_stopped_note(stopped)} The dataset is skipped: it "
                  "is written with every labeling the source has, or not at all.")
        return None

    def fits(labels: np.ndarray) -> bool:
        return labels.ndim == 1 and len(labels) == len(data)

    dropped = [f"{n} ({len(labels)} values)" if labels.ndim == 1 else f"{n} (shape {labels.shape})"
               for n, labels in labeling.items() if not fits(labels)]
    if dropped:
        log.warning(f"{dataset_group}/{dataset_name}: dropped {', '.join(dropped)}, since the "
                    f"data has {len(data)} rows, one label each; continuing with "
                    f"{len(labeling) - len(dropped)} labeling(s).")
    labeling = {n: labels for n, labels in labeling.items() if fits(labels)}
    if "labels0" not in labeling:
        log.error(f"No usable labels0 for {dataset_group}/{dataset_name}.")
        return None

    feature_cols = [f"Feature_{i + 1}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feature_cols)
    for name, labels in labeling.items():
        df[f"{GROUND_TRUTH_PREFIX}{name}"] = labels
    df[COHORT_COLUMN] = labeling["labels0"]

    noise = {name: int(np.sum(labels == 0)) for name, labels in labeling.items()}
    noted = _noise_note(noise) if any(noise.values()) else ""
    log.info(f"Loaded '{dataset_group}/{dataset_name}': {df.shape[0]} rows{noted}, "
             f"{len(feature_cols)} features, {len(labeling)} labeling(s).")
    return df


def _noise_note(noise: Mapping[str, int]) -> str:
    """Where the labelings mark noise points: clustbench gives them cluster id 0,
    which a run reads as one more cluster."""
    marked = ", ".join(f"{count} in {name}" for name, count in noise.items() if count)
    return f", noise points with cluster id 0 ({marked}), which a run reads as one more cluster"


SOURCE: Final[Source] = Source(
    key="clustbench",
    display="Clustbench",
    blurb="real benchmark datasets downloaded online (Gagolewski suite)",
    metadata=CLUSTBENCH_METADATA,
    datasets=CLUSTBENCH_DATASETS,
    heavy=CLUSTBENCH_HEAVY,
    seeded=False,
    suite_keys={"base_url": "text"},
    fetch=lambda battery, dataset, seed, **kw: fetch_clustbench_data(
        dataset_group=battery, dataset_name=dataset, **kw),
)
