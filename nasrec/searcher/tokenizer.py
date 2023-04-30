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

# Project module imports.
from copy import deepcopy
from typing import (
    List,
    Dict,
    Any,
)

import numpy as np

from nasrec.supernet.supernet import SuperNet


class Tokenizer(object):
    """
    Tokenizer to tokenize choice blocks into tokens.
    Buckets need to be allocated to materizalize symbols that represents an arch in the search space.

    For each supernet block, we use the following encoding scheme:
    [#choice_blocks] bits: binary encoding of whether use the conenction from a specific choice block (macro::dense_idx)
    [#choice_blocks] bits: binary encoding of whether use the conenction from a specific choice block (macro::sparse_idx)
    [#choice_blocks] bits: binary encoding of whether use the conenction from a specific choice block (macro::dense_left_idx)
    [#choice_blocks] bits: binary encoding of whether use the conenction from a specific choice block (macro::dense_right_idx)
    [#blocks] bits: binary encoding of whether use this block or not. (micro::active_nodes)
    [#dims] bits: one-hot encoding indicating which dimension to use. (micro::in_dims)
    Example choice block:
        choice = {
        'micro': [
            {'active_nodes': [0, 5], 'in_dims': 512},
            {'active_nodes': [0, 5], 'in_dims': 256},
            {'active_nodes': [0, 5], 'in_dims': 64},
            {'active_nodes': [0, 5], 'in_dims': 16},
            {'active_nodes': [2, 5], 'in_dims': 351},
            {'active_nodes': [0, 5], 'in_dims': 512},
            {'active_nodes': [0, 5], 'in_dims': 256}
        ],
        'macro': [
            {'dense_idx': [0],
             'sparse_idx': [0],
             'dense_left_idx': [1],
             'dense_right_idx': [1]},

            {'dense_idx': [2],
             'sparse_idx': [0],
             'dense_left_idx': [2],
             'dense_right_idx': [2]},

            {'dense_idx': [3],
             'sparse_idx': [0],
             'dense_left_idx': [3],
             'dense_right_idx': [3]},

            {'dense_idx': [4],
             'sparse_idx': [0],
             'dense_left_idx': [4],
             'dense_right_idx': [4]},

            {'dense_idx': [5],
             'sparse_idx': [0],
             'dense_left_idx': [5],
             'dense_right_idx': [5]},

            {'dense_idx': [5, 6],
             'sparse_idx': [0],
             'dense_left_idx': [6],
             'dense_right_idx': [6]},

            {'dense_idx': [7],
             'sparse_idx': [0],
             'dense_left_idx': [7],
             'dense_right_idx': [7]}
             ]
        }

    Args:
        num_blocks (int): Number of blocks.
        ops_config (dict or list): operation configuration to represent the search space. If 'ops_config' is a dict,
        it represents a generic search space configuration that is applied to all 'num_blocks' choice blocks.
        If 'ops_config' is a list, it should have exactly the same length as 'num_blocks' and sets up the search space
        configuration for each choice block.
    """

    def __init__(
        self,
        num_blocks: int,
        ops_config: Any,
    ):
        self._num_blocks = num_blocks
        self._ops_config = ops_config
        if isinstance(self._ops_config, list):
            self._num_nodes = [config["num_nodes"] for config in self._ops_config]
            self._sparse_node_dims = [config["sparse_node_dims"] for config in self._ops_config]
            self._sparse_node_dims_dict = [
                self._create_encoding_mapper(sparse_node_dims) for sparse_node_dims in self._sparse_node_dims
            ]
            self._dense_node_dims = [config["dense_node_dims"] for config in self._ops_config]
            self._dense_node_dims_dict = [
                self._create_encoding_mapper(dense_node_dims) for dense_node_dims in self._dense_node_dims
            ]
        else:
            self._num_nodes = ops_config["num_nodes"]
            self._sparse_node_dims = ops_config["sparse_node_dims"]
            self._sparse_node_dims_dict = self._create_encoding_mapper(self._sparse_node_dims)
            self._dense_node_dims = ops_config["dense_node_dims"]
            self._dense_node_dims_dict = self._create_encoding_mapper(self._dense_node_dims)

    @staticmethod
    def _create_encoding_mapper(collections: List[Any]):
        encoding_dict = {}
        for idx, item in enumerate(collections):
            encoding_dict[item] = idx
        return encoding_dict

    @staticmethod
    def _one_hot(a: np.ndarray, num_classes: int):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def _encode_active_nodes(
        self, active_node_indices: List[Any], num_nodes: int
    ) -> List[Any]:
        enc = []
        for i in range(num_nodes):
            if i in active_node_indices:
                enc.append(1)
            else:
                enc.append(0)
        return enc

    def _encode_choice_blocks(self, choice_block_indices: List[Any]) -> List[Any]:
        enc = []
        for i in range(self._num_blocks):
            if i in choice_block_indices:
                enc.append(1)
            else:
                enc.append(0)
        return enc

    def tokenize(self, choice: Dict[(Any, Any)]) -> List[Any]:
        full_enc = []
        # First, encode micro settings.
        for i in range(len(choice["macro"])):
            full_enc += self._encode_choice_blocks(choice["macro"][i]["dense_idx"])
            full_enc += self._encode_choice_blocks(choice["macro"][i]["sparse_idx"])
            full_enc += self._encode_choice_blocks(choice["macro"][i]["dense_left_idx"])
            full_enc += self._encode_choice_blocks(
                choice["macro"][i]["dense_right_idx"]
            )
        # Next, encode micro settings.
        for i in range(len(choice["micro"])):
            if isinstance(self._ops_config, list):
                num_nodes = self._num_nodes[i]
                dense_node_dims_dict = self._dense_node_dims_dict[i]
                sparse_node_dims_dict = self._sparse_node_dims_dict[i]
            else:
                num_nodes = self._num_nodes
                dense_node_dims_dict = self._dense_node_dims_dict
                sparse_node_dims_dict = self._sparse_node_dims_dict
            full_enc += self._encode_active_nodes(
                choice["micro"][i]["active_nodes"], num_nodes
            )
            full_enc += [dense_node_dims_dict[choice["micro"][i]["dense_in_dims"]]]
            full_enc += [sparse_node_dims_dict[choice["micro"][i]["sparse_in_dims"]]]
            full_enc = full_enc + [1, 0] if choice["micro"][i]['dense_sparse_interact'] == 0 else full_enc + [0, 1]
            full_enc = full_enc + [1, 0] if choice["micro"][i]['deep_fm'] == 0 else full_enc + [0, 1]

        return np.asarray(full_enc, dtype=np.int)

    def hash_token(self, token: List[Any]) -> str:
        str_token = [str(x) for x in token]
        return "".join(str_token)

    def mutate_spec(self, choice: Dict[(Any, Any)]) -> Dict[(Any, Any)]:
        # Mutate the spec by resampling. Note: this should be consistent with the way of sampling supernet.
        # Change this if you want.
        block_idx = np.random.choice(self._num_blocks)
        level_choice = "macro" if np.random.random() > 0.5 else "micro"
        mutated_choice = deepcopy(choice)
        if level_choice == "macro":
            # Do we need to force replace=False? replace=True gives more flexibility.
            # Note. We sample a maximum of 4 connections for each choice block. This should be consistent with
            # the definitions defined in nasrec/supernet/utils.py:_get_random_choice_vanilla().
            # Note: this feature is only enabled in the 0818-v6 search space.
            num_max_items_in_dense = 1 + np.random.choice(min(4, block_idx + 1))
            num_max_items_in_sparse = 1 + np.random.choice(min(4, block_idx + 1))
            # Uncomment to reproduce results in 0810-v5 search space.
            # num_max_items_in_dense = 1 + np.random.choice(block_idx + 1)
            # num_max_items_in_sparse = 1 + np.random.choice(block_idx + 1)
            num_dense_unique_bi_choices = 1
            dense_unique_bi_choices = np.random.choice(
                block_idx + 1, num_dense_unique_bi_choices * 2
            )
            new_macro_choice = {
                "dense_idx": np.random.choice(
                    block_idx + 1, num_max_items_in_dense, replace=False
                )
                .reshape(-1)
                .tolist(),
                "sparse_idx": np.random.choice(
                    block_idx + 1, num_max_items_in_sparse, replace=False
                )
                .reshape(-1)
                .tolist(),
                "dense_left_idx": dense_unique_bi_choices[:num_dense_unique_bi_choices]
                .reshape(-1)
                .tolist(),
                "dense_right_idx": dense_unique_bi_choices[num_dense_unique_bi_choices:]
                .reshape(-1)
                .tolist(),
            }
            macro_key = np.random.choice(
                ["dense_idx", "sparse_idx", "dense_left_idx", "dense_right_idx"]
            )
            mutated_choice["macro"][block_idx][macro_key] = deepcopy(
                new_macro_choice[macro_key]
            )
        else:
            ops_config = (
                self._ops_config[block_idx]
                if isinstance(self._ops_config, list)
                else self._ops_config
            )
            dense_node_dims = (
                self._dense_node_dims[block_idx]
                if isinstance(self._ops_config, list)
                else self._dense_node_dims
            )
            sparse_node_dims = (
                self._sparse_node_dims[block_idx]
                if isinstance(self._ops_config, list)
                else self._sparse_node_dims
            )
            new_micro_choice = {}
            while True:
                new_micro_choice = {
                    "active_nodes": sorted(
                        [np.random.choice(ops_config["dense_nodes"])]
                        + [np.random.choice(ops_config["sparse_nodes"])]
                    ),
                    "dense_in_dims": np.random.choice(dense_node_dims),
                    "sparse_in_dims": np.random.choice(sparse_node_dims),
                    'dense_sparse_interact': np.random.choice([0, 1]),
                    'deep_fm': np.random.choice([0, 1]),
                }
                if new_micro_choice["active_nodes"] != ops_config["zero_nodes"]:
                    break
            micro_key = np.random.choice(["active_nodes", "dense_in_dims", "sparse_in_dims", "dense_sparse_interact", 'deep_fm'])
            mutated_choice["micro"][block_idx][micro_key] = new_micro_choice[micro_key]

        return mutated_choice

    def generate_random_choice(self):
        """
        Generate a random choice from the tokenizer.
        """

        block_idx = np.random.choice(self._num_blocks)
        choice = {
            'macro': [],
            'micro': [],
        }
        for block_idx in range(self._num_blocks):
            num_max_items_in_dense = 1 + np.random.choice(min(4, block_idx + 1))
            num_max_items_in_sparse = 1 + np.random.choice(min(4, block_idx + 1))
            # Uncomment to reproduce results in 0810-v5 search space.
            # num_max_items_in_dense = 1 + np.random.choice(block_idx + 1)
            # num_max_items_in_sparse = 1 + np.random.choice(block_idx + 1)
            num_dense_unique_bi_choices = 1
            dense_unique_bi_choices = np.random.choice(
                block_idx + 1, num_dense_unique_bi_choices * 2
            )
            new_macro_choice = {
                "dense_idx": np.random.choice(
                    block_idx + 1, num_max_items_in_dense, replace=False
                )
                .reshape(-1)
                .tolist(),
                "sparse_idx": np.random.choice(
                    block_idx + 1, num_max_items_in_sparse, replace=False
                )
                .reshape(-1)
                .tolist(),
                "dense_left_idx": dense_unique_bi_choices[:num_dense_unique_bi_choices]
                .reshape(-1)
                .tolist(),
                "dense_right_idx": dense_unique_bi_choices[num_dense_unique_bi_choices:]
                .reshape(-1)
                .tolist(),
            }
            ops_config = (
                self._ops_config[block_idx]
                if isinstance(self._ops_config, list)
                else self._ops_config
            )
            dense_node_dims = (
                self._dense_node_dims[block_idx]
                if isinstance(self._ops_config, list)
                else self._dense_node_dims
            )
            sparse_node_dims = (
                self._sparse_node_dims[block_idx]
                if isinstance(self._ops_config, list)
                else self._sparse_node_dims
            )
            new_micro_choice = {}
            while True:
                new_micro_choice = {
                    "active_nodes": sorted(
                        [np.random.choice(ops_config["dense_nodes"])]
                        + [np.random.choice(ops_config["sparse_nodes"])]
                    ),
                    "dense_in_dims": np.random.choice(dense_node_dims),
                    "sparse_in_dims": np.random.choice(sparse_node_dims),
                    'dense_sparse_interact': np.random.choice([0, 1]),
                    'deep_fm': np.random.choice([0, 1]),
                }
                if new_micro_choice["active_nodes"] != ops_config["zero_nodes"]:
                    break
            choice['macro'].append(new_macro_choice)
            choice['micro'].append(new_micro_choice)
        return choice
