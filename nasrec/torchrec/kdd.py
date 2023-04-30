"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import torch.utils.data.datapipes as dp
from nasrec.torchrec.utils import (
    LoadFiles,
    ReadLinesFromCSV,
    safe_cast,
)
from torch.utils.data import IterDataPipe

# Pseudo-dense feature with all zeros.
INT_FEATURE_COUNT = 3
CAT_FEATURE_COUNT = 10
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

COLUMN_TYPE_CASTERS: List[Callable[[Union[int, str]], Union[int, str]]] = [
    lambda val: safe_cast(val, int, 0),
    *(lambda val: safe_cast(val, int, 0) for _ in range(INT_FEATURE_COUNT)),
    *(lambda val: safe_cast(val, str, "") for _ in range(CAT_FEATURE_COUNT)),
]


def _default_row_mapper(example: List[str]) -> Dict[str, Union[int, str]]:
    column_names = reversed(DEFAULT_COLUMN_NAMES)
    column_type_casters = reversed(COLUMN_TYPE_CASTERS)
    return {
        next(column_names): next(column_type_casters)(val) for val in reversed(example)
    }

def _kdd(
    paths: Iterable[str],
    *,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    **open_kw,
) -> IterDataPipe:
    datapipe = LoadFiles(paths, mode="r", **open_kw)
    datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
    if row_mapper:
        datapipe = dp.iter.Mapper(datapipe, row_mapper)
    return datapipe

def kdd_kaggle(
    path: str,
    *,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    **open_kw,
) -> IterDataPipe:
    """`Kaggle/Avazu Display Advertising <https://www.kaggle.com/c/criteo-display-ad-challenge/>`_ Dataset
    Args:
        root (str): local path to train or test dataset file.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split TSV line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example:
        >>> train_datapipe = criteo_kaggle(
        >>>     "/home/datasets/criteo_kaggle/train.txt",
        >>> )
        >>> example = next(iter(train_datapipe))
        >>> test_datapipe = criteo_kaggle(
        >>>     "/home/datasets/criteo_kaggle/test.txt",
        >>> )
        >>> example = next(iter(test_datapipe))
    """
    return _kdd((path,), row_mapper=row_mapper, **open_kw)
