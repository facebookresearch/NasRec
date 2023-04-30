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

# Important imports
import numpy as np


def _get_random_choice_vanilla(num_items, max_items=4):
    """
    Get the random choice from a few items, with equal probability. This guarantees that each part of the network
    is trained uniformally. (i.e., uniform sampling.)
    We cap the number of maximum items during sampling to avoid complicated networks.
    """
    return np.random.choice(min(num_items, max_items)) + 1


def _get_binomial_random_choice_with_expectation(num_items, p=0.5, max_items=4):
    """
    Get the random choice from a few items, with equal probability and a given distribution.
    We cap the number of maximum items during sampling to avoid complicated networks.
    """
    return 1 + np.random.binomial(min(num_items - 1, max_items - 1), p)


anypath_choice_fn = {
    "uniform": lambda num_items: _get_random_choice_vanilla(num_items, max_items=4),
    "binomial-0.5": lambda num_items: _get_binomial_random_choice_with_expectation(
        num_items, p=0.5, max_items=4
    ),
}


def assert_valid_ops_config(ops_config):
    for key in ops_config.keys():
        ops_config_standalone = ops_config[key]
        if isinstance(ops_config_standalone, list):
            for idx in range(len(ops_config_standalone)):
                assert ops_config_standalone[idx]["num_nodes"] == len(
                    ops_config_standalone[idx]["node_names"]
                ), ValueError(
                    "Number of nodes per config should be equivalent to the number of modules (node names) per config."
                )
        else:
            assert ops_config_standalone["num_nodes"] == len(
                ops_config_standalone["node_names"]
            ), ValueError(
                "Number of nodes per config should be equivalent to the number of modules (node names) per config."
            )
