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
    Optional
)

import pickle
import json
import os
import torch
import torch.nn as nn

def load_json(json_file_name: Optional[str] = None):
    assert json_file_name is not None, "Json file name should not be 'None'!"
    with open(json_file_name, 'r') as fp:
        json_file = json.load(fp)
    return json_file


def dump_json(json_file_name: Optional[str], data: Any):
    assert json_file_name is not None, "Json file name should not be 'None'!"
    with open(json_file_name, 'w') as fp:
        json.dump(data, fp)


def create_dir(dir_name: Optional[str] = None):
    assert dir_name is not None, "Directory name should not be 'None'!"
    os.makedirs(dir_name, exist_ok=True)


def dump_pickle_data(dump_path: Optional[str], data: Any):
    assert dump_path is not None, "Dump path should not be 'None'!"
    with open(dump_path, 'wb') as fp:
        pickle.dump(data, fp)


def load_pickle_data(load_path: Optional[str]):
    assert load_path is not None, "Load path should not be 'None'!"
    with open(load_path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def load_model_checkpoint(load_path: Optional[str]):
    assert load_path is not None, "Load model path should not be 'None'!"
    print("Loading weights from {}!".format(load_path))
    with open(load_path, "rb") as h_file:
        checkpoint = torch.load(h_file, map_location=torch.device("cpu"))
    return checkpoint


def save_model_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint.update({"optimizer_state_dict": optimizer.state_dict()})
    with open(save_path, "wb") as h_file:
        torch.save(checkpoint, h_file)
    print("Saved weights to {}!".format(save_path))

