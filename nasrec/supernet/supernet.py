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

# System Imports
import copy
from itertools import combinations
from typing import (
    List,
    Any,
    Union,
    Optional,
)

# Important Imports
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import sparse

# Project Imports
from nasrec.supernet.modules import (
    ElasticLinear,
    ElasticLinear3D,
    DotProduct,
    Sum,
    SigmoidGating,
    Transformer,
    Zeros3D,
    CleverMaskGenerator,
    CleverZeroTensorGenerator,
    Zeros2D,
    FactorizationMachine3D,
)
from nasrec.supernet.utils import (
    anypath_choice_fn,
    assert_valid_ops_config,
)
from nasrec.utils.config import NUM_EMBEDDINGS_CRITEO

_node_choices = {
    # Dense op
    "linear-2d": lambda use_layernorm, max_dims_or_dims, activation, fixed: ElasticLinear(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        activation=activation,
        fixed=fixed,
    ),
    "zeros-2d": lambda use_layernorm, max_dims_or_dims, activation, fixed: Zeros2D(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        activation=activation,
        fixed=fixed,
    ),
    "sigmoid-gating": lambda use_layernorm, max_dims_or_dims, activation, fixed: SigmoidGating(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        activation=activation,
        fixed=fixed,
    ),
    "sum": lambda use_layernorm, max_dims_or_dims, activation, fixed: Sum(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        activation=activation,
        fixed=fixed,
    ),
    # Dense-Sparse op
    "dot-product": lambda use_layernorm, max_dims_or_dims, embedding_dim, fixed: DotProduct(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        embedding_dim=embedding_dim,
        fixed=fixed,
    ),
    # Sparse-op
    "zeros-3d": lambda use_layernorm, max_dims_or_dims, activation, embedding_dim, fixed: Zeros3D(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        fixed=fixed,
        embedding_dim=embedding_dim,
        activation=activation,
    ),
    "transformer": lambda use_layernorm, max_dims_or_dims, activation, embedding_dim, fixed: Transformer(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        fixed=fixed,
        embedding_dim=embedding_dim,
        activation=activation,
    ),
    "linear-3d": lambda use_layernorm, max_dims_or_dims, activation, embedding_dim, fixed: ElasticLinear3D(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        activation=activation,
        embedding_dim=embedding_dim,
        fixed=fixed,
    ),
    "fm": lambda use_layernorm, max_dims_or_dims, fixed: FactorizationMachine3D(
        use_layernorm=use_layernorm,
        max_dims_or_dims=max_dims_or_dims,
        fixed=fixed,
    ),
}

# Name of nodes that requires one dense (2d) input.
_dense_unary_nodes = ["linear-2d", "zeros-2d"]
# Name of nodes that requires two dense (2d) input.
_dense_binary_nodes = ["sum", "sigmoid-gating"]
# Name of nodes that requires one dense (2d) input and one sparse input.
_dense_sparse_nodes = ["dot-product"]
# Name of nodes that requires one sparse (2d) input.
_sparse_nodes = ["zeros-3d", "transformer", "linear-3d"]


"""
Operation config that represents the whole search space.
'num_nodes': Number of nodes (modules) per choice block.
'node_names': Name of each node (module) per choice block.
'node_dims': Node dimensions (width) per choice block.
'dense_nodes': Number of nodes that responsible for dense features (with optinally sparse feature input).
'sparse_nodes': Number of nodes that responsible for sparse features (with optinally sparse feature input).
'zero_nodes': Nodes that produce zero output.
"""
ops_config_lib = {
    "xlarge": {
        "num_nodes": 6,
        "node_names": [
            "linear-2d",
            "dot-product",
            "sigmoid-gating",
            "sum",
            "transformer",
            "linear-3d",
        ],
        "dense_node_dims": [16, 32, 64, 128, 256, 512, 768, 1024],
        "sparse_node_dims": [16, 32, 48, 64],
        "dense_nodes": [0, 1, 2, 3],
        "sparse_nodes": [4, 5],
        "zero_nodes": [],
    },   
    "xlarge-zeros": {
        "num_nodes": 8,
        "node_names": [
            "linear-2d",
            "dot-product",
            "sigmoid-gating",
            "sum",
            "zeros-2d",
            "transformer",
            "zeros-3d",
            "linear-3d",
        ],
        "dense_node_dims": [16, 32, 64, 128, 256, 512, 768, 1024],
        "sparse_node_dims": [16, 32, 48, 64],
        "dense_nodes": [0, 1, 2, 3, 4],
        "sparse_nodes": [5, 6, 7],
        "zero_nodes": [4, 6],
    },
    "autoctr": {
        "num_nodes": 3,
        "node_names": ["linear-2d", "dot-product", "linear-3d"],
        "dense_node_dims": [16, 32, 64, 128, 256, 512, 768, 1024],
        "sparse_node_dims": [16, 32, 48, 64],
        "dense_nodes": [0, 1],
        "sparse_nodes": [2],
        "zero_nodes": [],
    },
}

# Need to check the ops configs to verfify some items: 1) 'num_nodes' == len('node_names'), ...
assert_valid_ops_config(ops_config_lib)

# Mask Generator for efficient paddings. If a padding exist,
# directly fetch it from CPU/GPU cache. Otherwise, generate it from scratch and copy to CPU/GPU.  
_zeros_generator = CleverZeroTensorGenerator()
_mask_generator = CleverMaskGenerator()

_default_path_sampling_strategy = {"macro": "any-path", "micro": "single-path"}

_single_path_sampling_strategy = {"macro": "single-path", "micro": "single-path"}

_any_path_sampling_strategy = {"macro": "any-path", "micro": "any-path"}

_full_path_sampling_strategy = {"macro": "full-path", "micro": "full-path"}

_fixed_path_sampling_strategy = {"macro": "fixed-path", "micro": "fixed-path"}

_evo_duoshot_sampling_strategy = {"macro": "evo-2shot-path", "micro": "evo-2shot-path"}

path_sampling_strategy_lib = {
    "default": _default_path_sampling_strategy,
    "single-path": _single_path_sampling_strategy,
    "any-path": _any_path_sampling_strategy,
    "full-path": _full_path_sampling_strategy,
    "fixed-path": _fixed_path_sampling_strategy,
    "evo-2shot-path": _evo_duoshot_sampling_strategy,
}


class SuperNet(nn.Module):
    """
    Top-level Supernet implementation.
    """
    def __init__(
        self,
        # Supernet configurations.
        num_blocks: int,
        ops_config: Any,
        use_layernorm: bool,
        activation: str = "relu",
        # Data configurations.
        num_embeddings: List[int] = NUM_EMBEDDINGS_CRITEO,
        sparse_input_size: int = 26,
        embedding_dim: int = 16,
        # Output configuration.
        last_n_blocks_out: int = 1,
        # Path sampling
        path_sampling_strategy: str = "default",
        fixed: bool = False,
        fixed_choice: Any = None,
        place_embedding_on_cpu: bool = False,
        anypath_choice: str = "uniform",
        supernet_training_steps: int = 0,
        candidate_choices: Optional[List] = None,
        use_final_sigmoid: bool = False,
    ):
        """
        Args:
            :params num_blocks (int): Number of block choices in the supernet.
            :params ops_config (dict): Config of the search space.
            :params use_layernorm (bool): Whether attach layer norm in different parts of the networks.
            :params activation (str): Can be one of ['relu', "identity"].
            :params last_n_blocks_out (int): The number of output blocks to concatenate when producing representations (
                i.e., tensor before the final logits)
            :params sparse_input_size (int): Number of sparse features to input.
            :params num_embeddings (int): Number of embeddings (tokens).
            :params embedding_dim (int): Dimension of embeddings.
            :params path_sampling_strategy (str): Can be one of ['default', 'single-path', 'full-path', 'any-path' or 'fixed-path'].
            :params fixed (bool): Whether use a fixed subnet setting (i.e., non-weight-sharing), or use a flexible subnet setting (
                i.e., with weight sharing.)
            :params fixed_choice: Under a fixed setting, we can provide a fixed choice to instantiate specific type of sub-networks.
            Otherwise, a random sub-network following 'path_sampling_strategy' will be sampled and generated.
            :params place_embedding_on_cpu: This will place the embedding layers on CPU to save memory.
            Warning: you should expect 10~100x slow down in training.
            :params anypath_choice(str): Choose the way to sample sub-networks when doing path sampling. Can be one of ['uniform', 'binomial-0.5'].
            :params supernet_training_steps (int): The number of initial steps that trains supernet. The probability will be linearly decayed.
            Only effective when path sampling strategy is in ['default', 'single-path' and 'any-path'].
        """
        super(SuperNet, self).__init__()
        assert num_blocks >= 1, ValueError(
            "Supernet must contain a minimum of 1 block, but found {}!".format(
                num_blocks
            )
        )
        self._num_blocks = num_blocks
        self._ops_config = ops_config
        self._use_layernorm = use_layernorm
        self._activation = activation
        # self._max_dims_or_dims = max(self._ops_config['node_dims'])
        self._last_n_blocks_out = last_n_blocks_out
        self._sparse_input_size = sparse_input_size
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        # Supernet macro path sampling strategy.
        self._path_sampling_strategy = path_sampling_strategy
        self._macro_path_sampling_strategy = path_sampling_strategy_lib[
            path_sampling_strategy]["macro"]
        self._candidate_choices = candidate_choices
        self._fixed = fixed
        self._embedding = self._embedding_layers(
            self._sparse_input_size, self._num_embeddings, self._embedding_dim
        )
        self._final = nn.LazyLinear(1)
        if use_final_sigmoid:
            self._final_sigmoid = nn.Sigmoid()
        else:
            self._final_sigmoid = None
        self._place_embedding_on_cpu = place_embedding_on_cpu
        self._supernet_training_steps = supernet_training_steps
        self._anypath_choice_fn = anypath_choice_fn[anypath_choice]
        # Initialize a training counter.
        self._supernet_train_steps_counter = -1
        # Initialize device args.
        self._device_args = None

        # assert sparse_input_size == len(NUM_EMBEDDINGS)

        # Initialize supernet blocks.
        self._blocks = nn.ModuleList([])

        if self._fixed and fixed_choice is not None:
            self.choice = fixed_choice
            # Last choice macro-level. Used for fixed path sampling.
            self.macro_last_choice = fixed_choice["macro"]
        else:
            # Record the choices for further processing.
            self.choice = []
            # Last choice macro-level. Used to control the same subnet for fixed path sampling.
            self.macro_last_choice = None

        # Assertion: 'fixed_path_strategy' should be explicitly specified for 'fixed' option.
        if self._fixed:
            assert self._macro_path_sampling_strategy == "fixed-path", ValueError(
                "'fixed_path_strategy' should be explicitly specified when 'fixed' option is True."
            )


        for idx in range(self._num_blocks):
            ops_config_supernet_block = (
                self._ops_config[idx]
                if isinstance(self._ops_config, list)
                else self._ops_config
            )
            max_dims_or_dims_dense = (
                max(self._ops_config[idx]["dense_node_dims"])
                if isinstance(self._ops_config, list)
                else max(self._ops_config["dense_node_dims"])
            )
            max_dims_or_dims_sparse = (
                max(self._ops_config[idx]["sparse_node_dims"])
                if isinstance(self._ops_config, list)
                else max(self._ops_config["sparse_node_dims"])
            )
            self._blocks.append(
                SuperNetBlock(
                    ops_config_supernet_block,
                    self._use_layernorm,
                    int(max_dims_or_dims_dense),
                    int(max_dims_or_dims_sparse),
                    self._embedding_dim,
                    self._activation,
                    path_sampling_strategy=path_sampling_strategy_lib[
                        path_sampling_strategy]["micro"],
                    fixed=self._fixed,
                    fixed_micro_choice=None
                    if (fixed_choice is None) or (not self._fixed)
                    else fixed_choice["micro"][idx],
                    anypath_choice=anypath_choice,
                    supernet_training_steps=self._supernet_training_steps,
                    sparse_input_size=self._sparse_input_size
                )
            )
        
        self._blocks = nn.ModuleList(self._blocks)


    def get_dense_parameters(self):
        params = []
        params += list(self._blocks.parameters())
        params += list(self._final.parameters())
        return params


    def get_sparse_parameters(self):
        params = list(self._embedding.parameters())
        return params

    def load_embeddings_from_dlrm(self, dlrm_ckpt_path=None):
        """
        Load embedding from a DLRM checkpoint. This is hard-coded as the DLRM arch is fixed.
        """
        if dlrm_ckpt_path is not None:
            print("Loading embedding layers from {}!".format(dlrm_ckpt_path))
            checkpoint = torch.load(dlrm_ckpt_path, map_location=torch.device("cpu"))
            assert "model_state_dict" in checkpoint.keys(), "Please use the DLRM checkpoint to load!"
            checkpoint = checkpoint['model_state_dict']
            for idx in range(len(self._embedding)):
                # Directly copy weight from loaded checkpoints.
                device = self._embedding[idx].weight.data.device
                self._embedding[idx].weight.data = checkpoint["embedding_layers.{}.weight".format(idx)]
                self._embedding[idx].weight.data = self._embedding[idx].weight.data.to(device)
            print("Done!") 
        pass

    def configure_path_sampling_strategy(self, strategy):
        assert strategy in [
            "full-path",
            "single-path",
            "any-path",
            "fixed-path",
            "evo-2shot-path",
            "default",
        ], "Strategy {} is not found!".format(strategy)
        self._path_sampling_strategy = strategy
        self._macro_path_sampling_strategy = path_sampling_strategy_lib[strategy][
            "macro"
        ]
        # Now, configure the strategy for all blocks.
        for block in self._blocks:
            block._micro_path_sampling_strategy = path_sampling_strategy_lib[strategy][
                "micro"
            ]

    def _embedding_layers(self, sparse_input_size, num_embeddings, embedding_dim):
        return torch.nn.ModuleList(
            [
                torch.nn.Embedding(num_embeddings[i], embedding_dim)
                for i in range(sparse_input_size)
            ]
        )

    def _input_stem_layers_bi_output(self, int_feats, cat_feats):
        """
        'int_feats': Integer features (dense features).
        'cat_feats': Categorical features (sparse features).
        """
        # Copy cat feats to CPU for embedding lookup. Should use .cpu() instead of .to(None) here to force cpu copy.
        if self._place_embedding_on_cpu:
            cat_feats = cat_feats.cpu()
        sparse_t_3d = torch.stack(
            [
                embedding_layer(cat_feats[:, idx])
                for idx, embedding_layer in enumerate(self._embedding)
            ],
            dim=1,
        )
        if self._place_embedding_on_cpu:
            sparse_t_3d = sparse_t_3d.to(int_feats.device)
        dense_t_2d = int_feats
        return dense_t_2d, sparse_t_3d

    def _get_choice(self):
        """
        Get a macro-level choice from the supernet. A choice looks like this (DLRM):
        [{"dense_idx": [0], "sparse_idx": [0], "dense_left_idx": [0], "dense_right_idx": [0]},
        {"dense_idx": [1], "sparse_idx": [0], "dense_left_idx": [1], "dense_right_idx": [1]},
        {"dense_idx": [2], "sparse_idx": [0], "dense_left_idx": [2], "dense_right_idx": [2]},
        {"dense_idx": [3], "sparse_idx": [0], "dense_left_idx": [3], "dense_right_idx": [3]},
        {"dense_idx": [4], "sparse_idx": [0], "dense_left_idx": [4], "dense_right_idx": [4]},
        {"dense_idx": [4, 5], "sparse_idx": [0], "dense_left_idx": [5], "dense_right_idx": [5]},
        {"dense_idx": [6], "sparse_idx": [0], "dense_left_idx": [6], "dense_right_idx": [6]}]
        Generally, "macro" denotes the input-level connections for choice blocks, where as 'micro' denotes the inner-block choice on which module to
        activate for the whole supernet.
        """
        # Use zero to avoid zero division.
        thresh = (
            1.0
            - self._supernet_train_steps_counter
            / (self._supernet_training_steps + 1e-10)
            if self._supernet_train_steps_counter < self._supernet_training_steps
            and self._supernet_train_steps_counter > 0
            else 0
        )
        if self._macro_path_sampling_strategy == "single-path":
            choice = (
                [self._get_full_path_choice(1 + idx) for idx in range(self._num_blocks)]
                if np.random.random() < thresh
                else [
                    self._get_single_path_choice(1 + idx)
                    for idx in range(self._num_blocks)
                ]
            )
        elif self._macro_path_sampling_strategy == "full-path":
            choice = [
                self._get_full_path_choice(1 + idx) for idx in range(self._num_blocks)
            ]
        elif self._macro_path_sampling_strategy == "any-path":
            choice = (
                [self._get_full_path_choice(1 + idx) for idx in range(self._num_blocks)]
                if np.random.random() < thresh
                else [
                    self._get_any_path_choice(1 + idx)
                    for idx in range(self._num_blocks)
                ]
            )
        elif (
            self._macro_path_sampling_strategy == "fixed-path"
            and self.macro_last_choice is None
        ):
            # Sanity check: fixed-path should be called only once!
            if hasattr(self, "__fixed_path_called"):
                raise ValueError(
                    "Error! fixed-path choice should be generated only once for each supernet!"
                )
            else:
                self.__fixed_path_called = True
            choice = [
                self._get_fixed_path_choice(1 + idx) for idx in range(self._num_blocks)
            ]
        elif self._macro_path_sampling_strategy == "fixed-path":
            choice = self.macro_last_choice
        elif self._macro_path_sampling_strategy == "evo-2shot-path":
            assert self._candidate_choices is not None, \
                "You must specify self._candidate_choices before using 'evo-2shot-path'!"
            choice_idx = np.random.randint(len(self._candidate_choices))
            choice = self._candidate_choices[choice_idx]['choice']
            self.choice["macro"] = choice["macro"]
            for i in range(self._num_blocks):
                self._blocks[i].configure_choice(choice["micro"][i])
            choice = choice['macro']
        else:
            raise NotImplementedError(
                "Path strategy {} is not supported!".format(
                    self._macro_path_sampling_strategy
                )
            )
        # Full-path sampling is usually needed for warm-up the whole supernet and do shape inference. Dis-regard full-path as a choice of arbitary path
        # for subnet allows more convenience when we need to fine-tune/evaluate sampled subnets.
        if self._macro_path_sampling_strategy != "full-path":
            self.macro_last_choice = choice
        return choice

    def forward(self, int_feats: torch.Tensor, cat_feats: torch.Tensor, choices=None):
        if self._fixed:
            return self.fixed_forward(int_feats, cat_feats, choices)
        # Increase counter for supernet.
        self._supernet_train_steps_counter += 1
        # Do forward.
        dense_t_2d, sparse_t_3d = self._input_stem_layers_bi_output(
            int_feats, cat_feats
        )
        dense_t_2d_list = [dense_t_2d]
        sparse_t_3d_list = [sparse_t_3d]
        self.choice = {"micro": [], "macro": []}
        if choices is None:
            choice = self._get_choice()
        else:
            choice = choices["macro"]
        self.choice["macro"] = choice
        for i in range(self._num_blocks):
            # Create all tensors needed.
            dense_t_2d_forward_list = []
            sparse_t_3d_forward_list = []
            dense_left_2d_forward_list = []
            dense_right_2d_forward_list = []
            for j in range(len(dense_t_2d_list)):
                if j in choice[i]["dense_idx"]:
                    dense_t_2d_forward_list.append(dense_t_2d_list[j])
                else:
                    dense_t_2d_forward_list.append(
                        _zeros_generator(
                            dense_t_2d_list[j].size(), dense_t_2d_list[j].device
                        )
                    )
                if j in choice[i]["sparse_idx"]:
                    sparse_t_3d_forward_list.append(sparse_t_3d_list[j])
                else:
                    sparse_t_3d_forward_list.append(
                        _zeros_generator(
                            sparse_t_3d_list[j].size(), sparse_t_3d_list[j].device
                        )
                    )
                if j in choice[i]["dense_left_idx"]:
                    dense_left_2d_forward_list.append(dense_t_2d_list[j])
                else:
                    dense_left_2d_forward_list.append(
                        _zeros_generator(
                            dense_t_2d_list[j].size(), dense_t_2d_list[j].device
                        )
                    )
                if j in choice[i]["dense_right_idx"]:
                    dense_right_2d_forward_list.append(dense_t_2d_list[j])
                else:
                    dense_right_2d_forward_list.append(
                        _zeros_generator(
                            dense_t_2d_list[j].size(), dense_t_2d_list[j].device
                        )
                    )
            # Now, cat torch tensors.
            dense_t_2d_forward = torch.cat(dense_t_2d_forward_list, dim=-1)
            sparse_t_3d_forward = torch.cat(sparse_t_3d_forward_list, dim=1)
            dense_left_2d_forward = torch.cat(dense_left_2d_forward_list, dim=-1)
            dense_right_2d_forward = torch.cat(dense_right_2d_forward_list, dim=-1)
            block_choice = None if choices is None else choices["micro"]
            # Forward 3D tensors
            dense_t_out_2d, sparse_t_out_3d = self._blocks[i](
                (
                    dense_t_2d_forward,
                    sparse_t_3d_forward,
                    dense_left_2d_forward,
                    dense_right_2d_forward,
                ),
                block_choice,
            )
            self.choice["micro"].append(self._blocks[i].choice)

            dense_t_2d_list.append(dense_t_out_2d)
            sparse_t_3d_list.append(sparse_t_out_3d)

        # Finally, get everything to 2D and reshape. We pass the output of last 1
        # blocks to get good trade-off between acc and resource. This is hard-coded.
        flattened_dense = torch.cat(dense_t_2d_list[-self._last_n_blocks_out :], dim=-1)
        
        flattened_sparse = torch.flatten(
            torch.cat(sparse_t_3d_list[-self._last_n_blocks_out:], dim=-1), 1, -1
        )
        feats = torch.cat([flattened_dense, flattened_sparse], dim=-1)
        out = self._final(feats)
        if self._final_sigmoid is not None:
            return self._final_sigmoid(out)
        else:
            return out

    @torch.jit.ignore
    def fixed_forward(
        self, int_feats: torch.Tensor, cat_feats: torch.Tensor, choices: Any
    ):
        dense_t_2d, sparse_t_3d = self._input_stem_layers_bi_output(
            int_feats, cat_feats
        )
        dense_t_2d_list = [dense_t_2d]
        sparse_t_3d_list = [sparse_t_3d]
        self.choice = {"micro": [], "macro": []}
        if choices is None:
            choice = self._get_choice()
        else:
            choice = choices["macro"]
        self.choice["macro"] = choice
        for i in range(self._num_blocks):
            # Create all tensors needed.
            dense_t_2d_forward_list = []
            sparse_t_3d_forward_list = []
            dense_left_2d_forward_list = []
            dense_right_2d_forward_list = []
            for j in range(len(dense_t_2d_list)):
                if j in choice[i]["dense_idx"]:
                    dense_t_2d_forward_list.append(dense_t_2d_list[j])
                if j in choice[i]["sparse_idx"]:
                    sparse_t_3d_forward_list.append(sparse_t_3d_list[j])
                if j in choice[i]["dense_left_idx"]:
                    dense_left_2d_forward_list.append(dense_t_2d_list[j])
                if j in choice[i]["dense_right_idx"]:
                    dense_right_2d_forward_list.append(dense_t_2d_list[j])
            # Now, cat torch tensors.
            dense_t_2d_forward = torch.cat(dense_t_2d_forward_list, dim=-1)
            sparse_t_3d_forward = torch.cat(sparse_t_3d_forward_list, dim=1)
            dense_left_2d_forward = torch.cat(dense_left_2d_forward_list, dim=-1)
            dense_right_2d_forward = torch.cat(dense_right_2d_forward_list, dim=-1)
            block_choice = None if choices is None else choices["micro"]
            # Forward 3D tensors
            dense_t_out_2d, sparse_t_out_3d = self._blocks[i](
                (
                    dense_t_2d_forward,
                    sparse_t_3d_forward,
                    dense_left_2d_forward,
                    dense_right_2d_forward,
                ),
                block_choice,
            )
            self.choice["micro"].append(self._blocks[i].choice)

            dense_t_2d_list.append(dense_t_out_2d)
            sparse_t_3d_list.append(sparse_t_out_3d)

        # Finally, get everything to 2D and reshape. We pass the output of last 1
        # blocks to get good trade-off between acc and resource. This is hard-coded.
        flattened_dense = torch.cat(dense_t_2d_list[-self._last_n_blocks_out :], dim=-1)
        
        flattened_sparse = torch.flatten(
            torch.cat(sparse_t_3d_list[-self._last_n_blocks_out:], dim=-1), 1, -1
        )
        
        feats = torch.cat([flattened_dense, flattened_sparse], dim=-1)
        out = self._final(feats)
        if self._final_sigmoid is not None:
            return self._final_sigmoid(out)
        else:
            return out

    def get_all_subnet_macro_choices(self, block_idx: int):
        max_items_in_dense_and_sparse = 1 + block_idx
        all_macro_choices = {
            "dense_left_idx": [],
            "dense_right_idx": [],
            "dense_idx": [],
            "sparse_idx": [],
        }
        for num_items_in_dense in range(1, max_items_in_dense_and_sparse + 1):
            dense_idx_lists = list(
                combinations(
                    list(range(max_items_in_dense_and_sparse)), num_items_in_dense
                )
            )
            all_macro_choices["dense_idx"] += dense_idx_lists

        for num_items_in_sparse in range(1, max_items_in_dense_and_sparse + 1):
            sparse_idx_lists = list(
                combinations(
                    list(range(max_items_in_dense_and_sparse)), num_items_in_sparse
                )
            )
            all_macro_choices["sparse_idx"] += sparse_idx_lists

        for num_dense_unique_bi_choices in range(
            1, min(2, max_items_in_dense_and_sparse + 1)
        ):
            dense_left_idx_lists = list(
                combinations(
                    list(range(max_items_in_dense_and_sparse)),
                    num_dense_unique_bi_choices,
                )
            )
            dense_right_idx_lists = list(
                combinations(
                    list(range(max_items_in_dense_and_sparse)),
                    num_dense_unique_bi_choices,
                )
            )
            all_macro_choices["dense_left_idx"] += dense_left_idx_lists
            all_macro_choices["dense_right_idx"] += dense_right_idx_lists

        return all_macro_choices

    def get_all_subnet_choices(self):
        all_choices = {"macro": [], "micro": []}
        for block_idx in range(self._num_blocks):
            all_choices["macro"].append(self.get_all_subnet_macro_choices(block_idx))
            all_choices["micro"].append(
                self._blocks[block_idx].get_all_subnet_micro_choices()
            )
        return all_choices

    def _get_single_path_choice(self, max_items_in_dense_and_sparse: int):
        """
        Single path sampling.
        Sample one of the single blocks ahead of the current choice block.
        """
        # 1 pair of gating here.
        dense_unique_bi_choices = np.random.choice(max_items_in_dense_and_sparse, 1 * 2)
        choice = {
            "dense_idx": [np.random.choice(max_items_in_dense_and_sparse)],
            "sparse_idx": [np.random.choice(max_items_in_dense_and_sparse)],
            "dense_left_idx": [dense_unique_bi_choices[0]],
            "dense_right_idx": [dense_unique_bi_choices[1]],
        }
        return choice

    def _get_any_path_choice(self, max_items_in_dense_and_sparse: int):
        """
        Any path sampling.
        Sample any of the previous blocks ahead of the current choice blocks.
        Different from '_get_fixed_path_choice' implemented next, this path sampling is affected by variable 'anypath_choice'.
        """
        num_items_in_dense = self._anypath_choice_fn(max_items_in_dense_and_sparse)
        num_items_in_sparse = self._anypath_choice_fn(max_items_in_dense_and_sparse)
        # Limit 1 maximum binary dense fusion here to save some memory.
        num_dense_unique_bi_choices = 1
        # num_dense_unique_bi_choices = max(1, (1 + np.random.choice(max_items_in_dense_and_sparse)) // 2)
        dense_unique_bi_choices = np.random.choice(
            max_items_in_dense_and_sparse, num_dense_unique_bi_choices * 2
        )
        choice = {
            "dense_idx": np.random.choice(
                max_items_in_dense_and_sparse, num_items_in_dense, replace=False
            )
            .reshape(-1)
            .tolist(),
            "sparse_idx": np.random.choice(
                max_items_in_dense_and_sparse, num_items_in_sparse, replace=False
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
        return choice

    def _get_fixed_path_choice(self, max_items_in_dense_and_sparse: int):
        """
        Fixed path sampling.
        Sample any of the previous blocks ahead of the current choice blocks.
        Different from '_get_any_path_choice' implemented next, this path sampling is NOT affected by variable 'anypath_choice'.
        """
        # Note. We sample a maximum of 4 connections for each choice block. This should be consistent with
        # the definitions defined in nasrec/supernet/utils.py:_get_random_choice_vanilla().
        # Note: this feature is only enabled in the 0818-v6 search space.
        num_items_in_dense = anypath_choice_fn["uniform"](max_items_in_dense_and_sparse)
        num_items_in_sparse = anypath_choice_fn["uniform"](
            max_items_in_dense_and_sparse
        )
        # Uncomment to use the 0810-v5 search space or before.
        # num_items_in_dense = 1 + np.random.choice(max_items_in_dense_and_sparse)
        # num_items_in_sparse = 1 + np.random.choice(max_items_in_dense_and_sparse)
        # Limit 1 maximum binary dense fusion here to save some memory.
        num_dense_unique_bi_choices = 1
        # num_dense_unique_bi_choices = max(1, (1 + np.random.choice(max_items_in_dense_and_sparse)) // 2)
        dense_unique_bi_choices = np.random.choice(
            max_items_in_dense_and_sparse, num_dense_unique_bi_choices * 2
        )
        choice = {
            "dense_idx": np.random.choice(
                max_items_in_dense_and_sparse, num_items_in_dense, replace=False
            )
            .reshape(-1)
            .tolist(),
            "sparse_idx": np.random.choice(
                max_items_in_dense_and_sparse, num_items_in_sparse, replace=False
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
        return choice

    def _get_full_path_choice(self, max_items_in_dense_and_sparse: int):
        """
        The full supernet. This is used to warmup a supernet model.
        """
        choice = {
            "dense_idx": np.arange(max_items_in_dense_and_sparse),
            "sparse_idx": np.arange(max_items_in_dense_and_sparse),
            "dense_left_idx": np.arange(max_items_in_dense_and_sparse),
            "dense_right_idx": np.arange(max_items_in_dense_and_sparse),
        }
        return choice

    def to(self, *args):
        if not self._place_embedding_on_cpu:
            return super(SuperNet, self).to(*args)
        for _, m in self.named_modules():
            # Skip self and ModuleList to avoid duplication.
            if isinstance(m, SuperNet) or isinstance(m, nn.ModuleList):
                continue
            # Copy all layers excluding embedding to GPU.
            m = (
                super(type(m), m).to(*args)
                if not isinstance(m, nn.Embedding)
                else super(type(m), m).to(None)
            )
        self._device_args = args
        return self

    def configure_choice(self, choice: Any):
        self.choice = copy.deepcopy(choice)
        # Configure macro last choice.
        self.macro_last_choice = copy.deepcopy(choice["macro"])
        # Configure the choice for each block.
        for idx in range(self._num_blocks):
            self._blocks[idx].configure_choice(choice["micro"][idx])

    def set_mode_to_finelune_last_only(self):
        self._embedding.requires_grad_(False)
        self._blocks.requires_grad_(False)
        self._final.requires_grad_(True)

    def set_mode_to_normal_mode(self):
        self._embedding.requires_grad_(True)
        self._blocks.requires_grad_(True)
        self._final.requires_grad_(True)

    def set_mode_to_layernorm_calibrate(self):
        self._embedding.requires_grad_(False)
        self._blocks.requires_grad_(False)
        self._final.requires_grad_(False)
        counter = 0
        for _, m in self._blocks.named_modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                counter += 1

    def set_mode_to_finetune_no_embedding(self):
        self._embedding.requires_grad_(False)
        self._blocks.requires_grad_(True)
        self._final.requires_grad_(True)

    def discretize_config_each_block(self, probs: Union[np.ndarray, List[Any]]):
        configs = []
        for idx in range(self._num_blocks):
            config_block = self._blocks[idx].discretize_config_each_block(probs[idx])
            configs.append(config_block)
        return configs

DS_INTERACT_NUM_SPLITS = 8

class SuperNetBlock(nn.Module):
    def __init__(
        self,
        ops_config: Any,
        use_layernorm: bool,
        max_dims_or_dims_dense: int,
        max_dims_or_dims_sparse: int,
        embedding_dim: int,
        activation: str = "relu",
        path_sampling_strategy: str = "single-path",
        fixed: bool = False,
        fixed_micro_choice=None,
        anypath_choice: str = "uniform",
        supernet_training_steps: int = 0,
        sparse_input_size: int = 26,

    ):
        """
        SuperNet Block. Each block will output a 2D tensor and a 3D tensor.
        Args:
            :params ops_config (dict): Configuration for operations.
            :params use_layernorm (bool): Whether use layernorm or not.
            :params max_dims_or_dims (int): Maximum number of dimensions.
            :params embedding_dim (intfixed=: Under a fixed setting, we can provide a fixed choice to instantiate specific type of sub-networks.
            Otherwise, a random sub-network following 'path_sampling_strategy' will be sampled and generated.
            :params supernet_block_training_prob (float): Probability of training the whole supernet.
            Only applicable in ['single-path', 'any-path'].
            :params anypath_choice(str): Choose the way to sample sub-networks when doing path sampling. Can be one of ['uniform', 'binomial-0.5'].
            :params supernet_training_steps (int): The number of initial steps that trains supernet. The probability will be linearly decayed.
            Only effective when path sampling strategy is in ['single-path' and 'any-path'].
        """
        super(SuperNetBlock, self).__init__()

        self._num_nodes = ops_config["num_nodes"]
        self._dense_nodes = ops_config["dense_nodes"]
        self._sparse_nodes = ops_config["sparse_nodes"]
        self._node_names = ops_config["node_names"]
        self._dense_node_dims = ops_config["dense_node_dims"]
        self._sparse_node_dims = ops_config["sparse_node_dims"]
        self._zero_nodes = ops_config["zero_nodes"]
        self._sparse_input_size = sparse_input_size
        self._use_layernorm = use_layernorm
        self._max_dims_or_dims_dense = max_dims_or_dims_dense
        self._max_dims_or_dims_sparse = max_dims_or_dims_sparse
        self._embedding_dim = embedding_dim
        self._activation = activation
        self._micro_path_sampling_strategy = path_sampling_strategy
        self._fixed = fixed
        self._fixed_micro_choice = fixed_micro_choice
        self._anypath_choice_fn = anypath_choice_fn[anypath_choice]
        self._supernet_training_steps = supernet_training_steps
        # Device placement.
        self._device_args = None
        # Initialize a training counter.
        self._supernet_train_steps_counter = -1

        # Now, initialize all nodes with parameters.
        self._nodes = nn.ModuleList()

        # Choice for block. Recorded for fixed path sampling.
        self.micro_last_choice = self._fixed_micro_choice if self._fixed else None
        if self._fixed:
            choice = self._get_choice()
            choice_nodes = choice["active_nodes"]
        else:
            choice_nodes = list(range(self._num_nodes))
        for i in range(self._num_nodes):
            if i not in choice_nodes:
                self._nodes.append(nn.ModuleList([]))
                continue
            if self._node_names[i] in _dense_binary_nodes + _dense_unary_nodes:
                node = _node_choices[self._node_names[i]](
                    self._use_layernorm,
                    choice["dense_in_dims"] if self._fixed else self._max_dims_or_dims_dense,
                    self._activation,
                    fixed=self._fixed,
                )
            elif self._node_names[i] in _sparse_nodes:
                node = _node_choices[self._node_names[i]](
                    self._use_layernorm,
                    choice["sparse_in_dims"] if self._fixed else self._max_dims_or_dims_sparse,
                    embedding_dim=self._embedding_dim,
                    fixed=self._fixed,
                    activation=self._activation,
                )
            elif self._node_names[i] in _dense_sparse_nodes:
                node = _node_choices[self._node_names[i]](
                    self._use_layernorm,
                    choice["dense_in_dims"] if self._fixed else self._max_dims_or_dims_dense,
                    self._embedding_dim,
                    fixed=self._fixed,
                )
            else:
                raise NotImplementedError(
                    "Block name {} is not supported in supernet!".format(
                        self._node_names[i]
                    )
                )
            self._nodes.append(node)


        # Dense-Sparse Interaction
        if self._fixed and choice['dense_sparse_interact'] == 0:
            self.project_emb_dim, self.project_emb_dim_layernorm = None, None
        else:
            self.ds_interact_expanded_dim = DS_INTERACT_NUM_SPLITS * self._embedding_dim
            # Initialize dense-to-sparse projection layers.
            self.project_emb_dim = nn.LazyLinear(
                self.ds_interact_expanded_dim, 
                bias=not use_layernorm)
            self.project_emb_dim_layernorm = nn.LayerNorm(
                self.ds_interact_expanded_dim, eps=1e-5) if use_layernorm else None
    
        # Sparse-Dense Interaction.
        if self._fixed and choice["deep_fm"] == 0:
            self.deep_fm, self.deep_fm_output_ln = None, None
        else:
            self.deep_fm_dims = max(self._dense_node_dims) if not self._fixed else choice['dense_in_dims']
            self.deep_fm = FactorizationMachine3D(
                fixed=self._fixed, use_layernorm=self._use_layernorm, max_dims_or_dims=self.deep_fm_dims)
            # Layernorm already contained.

        # Record choices.
        self.choice = []

    def _get_choice(self):
        """
        Get a micro-level choice which looks like this:
        {"active_nodes": [0, 6], "in_dims": 512}
        """
        thresh = (
            1.0 - self._supernet_train_steps_counter
            / (self._supernet_training_steps + 1e-10)
            if self._supernet_train_steps_counter < self._supernet_training_steps
            and self._supernet_train_steps_counter > 0
            else 0
        )
        if self._micro_path_sampling_strategy == "single-path":
            choice = (
                self._get_full_path_choice()
                if np.random.random() < thresh
                else self._get_single_path_choice()
            )
        elif self._micro_path_sampling_strategy == "full-path":
            choice = self._get_full_path_choice()
        elif self._micro_path_sampling_strategy == "any-path":
            choice = (
                self._get_full_path_choice()
                if np.random.random() < thresh
                else self._get_any_path_choice()
            )
        elif (
            self._micro_path_sampling_strategy == "fixed-path"
            and self.micro_last_choice is None
        ):
            # Sanity check: fixed-path should be called only once!
            if hasattr(self, "__fixed_path_called"):
                raise ValueError(
                    "Error! fixed-path choice should be generated only once for each supernet!"
                )
            else:
                self.__fixed_path_called = True
            choice = self._get_fixed_path_choice()
        elif self._micro_path_sampling_strategy == "fixed-path":
            choice = self.micro_last_choice
        elif self._micro_path_sampling_strategy == "evo-2shot-path":
            # Do nothing as the choice has been configured.
            choice = self.micro_last_choice
        else:
            raise NotImplementedError(
                "Path strategy {} is not supported!".format(
                    self._micro_path_sampling_strategy
                )
            )
        if self._micro_path_sampling_strategy != "full-path":
            self.micro_last_choice = choice

        return choice

    """
    Forward function. Assume that random masking has been applied in SuperNet and we get the masked input.
    All dense tensors 'dense_t' and sparse tensors 'sparse_t' with choice to proceed.
    """
    def forward(self, tensors, choices=None):
        if choices is None:
            choice = self._get_choice()
        else:
            choice = choices
        self.choice = choice
        if self._fixed:
            return self.fixed_forward(tensors, choice)
        # Increase counter for supernet.
        self._supernet_train_steps_counter += 1
        # Do forward.
        dense_t_2d, sparse_t_3d, dense_left_2d, dense_right_2d = tensors
        output_2d = []
        output_3d = []
        for i in range(self._num_nodes):
            out_2d, out_3d = None, None
            # Case 1: encounter an inactive block in 2D.
            if i not in choice["active_nodes"] and (
                self._node_names[i]
                in _dense_sparse_nodes + _dense_binary_nodes + _dense_unary_nodes
            ):
                # Zeroing nodes that are not very effective.
                zeros_2d = _zeros_generator(
                    torch.Size((dense_t_2d.size(0), self._max_dims_or_dims_dense)),
                    dense_t_2d.device,
                )
                output_2d.append(zeros_2d)
                continue
    
            if i not in choice["active_nodes"] and (
                self._node_names[i] in _sparse_nodes
            ):
                # Zeroing nodes that are not very effective.
                zeros_3d = _zeros_generator(
                    torch.Size(
                        (
                            sparse_t_3d.size(0),
                            self._max_dims_or_dims_sparse,
                            sparse_t_3d.size(2),
                        )
                    ),
                    sparse_t_3d.device,
                )
                output_3d.append(zeros_3d)
                continue

            if self._node_names[i] in _dense_unary_nodes:
                out_2d = self._nodes[i](dense_t_2d, choice["dense_in_dims"])
            elif self._node_names[i] in _dense_sparse_nodes:
                out_2d = self._nodes[i](dense_t_2d, sparse_t_3d, choice["dense_in_dims"])
            elif self._node_names[i] in _dense_binary_nodes:
                out_2d = self._nodes[i](
                    dense_left_2d, dense_right_2d, choice["dense_in_dims"]
                )
            elif self._node_names[i] in _sparse_nodes:
                out_3d = self._nodes[i](sparse_t_3d, choice["sparse_in_dims"])
            else:
                raise NotImplementedError(
                    "Block name {} is not supported!".format(self._node_names[i])
                )
            if out_2d is not None:
                output_2d.append(out_2d)
            if out_3d is not None:
                output_3d.append(out_3d)


        dense_t_2d_out = torch.sum(torch.stack(output_2d, dim=-1), dim=-1)
        sparse_t_3d_out = torch.sum(torch.stack(output_3d, dim=-1), dim=-1)

        # Dense Sparse Interaction
        if choice['dense_sparse_interact'] == 1:
            if dense_t_2d_out.size(-1) != self._embedding_dim * DS_INTERACT_NUM_SPLITS:
                dense_t_2d_out_proj = dense_t_2d_out.clone()
                dense_t_2d_out_proj = self.project_emb_dim(dense_t_2d_out_proj)
                dense_t_2d_out_proj = self.project_emb_dim_layernorm(dense_t_2d_out_proj) \
                    if self._use_layernorm else dense_t_2d_out_proj
            else:
                self.project_emb_dim, self.project_emb_dim_layernorm = None, None
                dense_t_2d_out_proj = dense_t_2d_out
            dense_t_2d_out_proj = dense_t_2d_out_proj.view([-1, DS_INTERACT_NUM_SPLITS, self._embedding_dim])
        elif choice['dense_sparse_interact'] == 0:
            zeros_tensor_size = torch.Size(
                    [sparse_t_3d.size(0), DS_INTERACT_NUM_SPLITS, self._embedding_dim])
            dense_t_2d_out_proj = _zeros_generator(zeros_tensor_size, dense_t_2d_out.device)
        else:
            raise NotImplementedError("Bug reported for dense/sparse interact.")

        if choice['deep_fm'] == 1:
            sparse_t_3d_out_to_dense = self.deep_fm(sparse_t_3d_out, choice['dense_in_dims'])
            # Apply DeepFM, and added to dense.
            dense_t_2d_out += sparse_t_3d_out_to_dense
    
        # Merge dense and sparse features.
        # dense_t_2d_out_to_sparse = torch.stack([dense_t_2d_out_proj] * sparse_t_3d_out.size(1), dim=1)
        sparse_t_3d_out = torch.cat([sparse_t_3d_out, dense_t_2d_out_proj], dim=1)
        return dense_t_2d_out, sparse_t_3d_out

    def get_all_subnet_micro_choices(self):
        all_micro_choices = {
            "active_nodes": [],
            "dense_in_dims": [],
            "sparse_in_dims": [],
            'dense_sparse_interact': [0, 1]
        }
        for sparse_nodes in self._sparse_nodes:
            for dense_nodes in self._dense_nodes:
                all_micro_choices["active_nodes"].append(
                    (
                        dense_nodes,
                        sparse_nodes,
                    )
                )
        for in_dims in self._dense_node_dims:
            all_micro_choices["dense_in_dims"].append((in_dims,))
        for in_dims in self._sparse_node_dims:
            all_micro_choices["sparse_in_dims"].append((in_dims,))
        return all_micro_choices

    def fixed_forward(self, tensors, choices):
        choice = choices
        self.choice = choice
        dense_t_2d, sparse_t_3d, dense_left_2d, dense_right_2d = tensors
        output_2d = []
        output_3d = []
        for i in range(self._num_nodes):
            out_2d, out_3d = None, None
            # Case 1: encounter an inactive block in 2D.
            if i not in choice["active_nodes"]:
                continue
            if self._node_names[i] in _dense_unary_nodes:
                out_2d = self._nodes[i](dense_t_2d, choice["dense_in_dims"])
            elif self._node_names[i] in _dense_sparse_nodes:
                out_2d = self._nodes[i](dense_t_2d, sparse_t_3d, choice["dense_in_dims"])
            elif self._node_names[i] in _dense_binary_nodes:
                out_2d = self._nodes[i](
                    dense_left_2d, dense_right_2d, choice["dense_in_dims"]
                )
            elif self._node_names[i] in _sparse_nodes:
                out_3d = self._nodes[i](sparse_t_3d, choice["sparse_in_dims"])
            else:
                raise NotImplementedError(
                    "Block name {} is not supported!".format(self._node_names[i])
                )
            if out_2d is not None:
                output_2d.append(out_2d)
            if out_3d is not None:
                output_3d.append(out_3d)

        dense_t_2d_out = torch.sum(torch.stack(output_2d, dim=-1), dim=-1)
        sparse_t_3d_out = torch.sum(torch.stack(output_3d, dim=-1), dim=-1)

        if choice['dense_sparse_interact'] == 1:
            if dense_t_2d_out.size(-1) != self._embedding_dim * DS_INTERACT_NUM_SPLITS:
                dense_t_2d_out_proj = dense_t_2d_out.clone()
                dense_t_2d_out_proj = self.project_emb_dim(dense_t_2d_out_proj)
                dense_t_2d_out_proj = self.project_emb_dim_layernorm(dense_t_2d_out_proj) \
                    if self._use_layernorm else dense_t_2d_out_proj
            else:
                self.project_emb_dim, self.project_emb_dim_layernorm = None, None
                dense_t_2d_out_proj = dense_t_2d_out
            dense_t_2d_out_proj = dense_t_2d_out_proj.view([-1, DS_INTERACT_NUM_SPLITS, self._embedding_dim])
        else:
            zeros_tensor_size = torch.Size(
                    [sparse_t_3d.size(0), DS_INTERACT_NUM_SPLITS, self._embedding_dim])
            dense_t_2d_out_proj = _zeros_generator(zeros_tensor_size, dense_t_2d_out.device)

        if choice['deep_fm'] == 1:
            sparse_t_3d_out_to_dense = self.deep_fm(sparse_t_3d_out, choice['dense_in_dims'])
            # Apply DeepFM, and added to dense.
            dense_t_2d_out += sparse_t_3d_out_to_dense

        # Merge dense and sparse features.
        if choice['dense_sparse_interact'] == 1:
            return dense_t_2d_out, torch.cat([sparse_t_3d_out, dense_t_2d_out_proj], dim=1)
        else:
            return dense_t_2d_out, sparse_t_3d_out

    def _get_single_path_choice(self):
        """
        Single path sampling.
        Sample one single node for the current choice block.
        """
        while True:
            choice = {
                "active_nodes": sorted(
                    [np.random.choice(self._dense_nodes)]
                    + [np.random.choice(self._sparse_nodes)]
                ),
                "dense_in_dims": np.random.choice(self._dense_node_dims),
                "sparse_in_dims": np.random.choice(self._sparse_node_dims),
                'dense_sparse_interact': np.random.choice([0, 1]),
                "deep_fm": np.random.choice([0, 1]),
            }
            # Add constraint: dense/sparse node cannot be both zeros.
            if choice["active_nodes"] != self._zero_nodes:
                break
        return choice

    def _get_full_path_choice(self):
        """
        Full path sampling. Enable all modules in the choice block.
        """
        choice = {
            "active_nodes": np.arange(self._num_nodes),
            "dense_in_dims": np.max(self._dense_node_dims),
            "sparse_in_dims": np.max(self._sparse_node_dims),
            'dense_sparse_interact': 1,
            "deep_fm": 1,
        }
        return choice

    def _get_any_path_choice(self):
        """
        Fixed path sampling.
        Sample any of the nodes for the current choice blocks.
        Different from '_get_any_path_choice' implemented next, this path sampling is affected by variable 'anypath_choice'.
        """
        while True:
            num_dense_nodes = self._anypath_choice_fn(len(self._dense_nodes))
            num_sparse_nodes = self._anypath_choice_fn(len(self._sparse_nodes))
            dense_nodes = np.random.choice(
                self._dense_nodes, num_dense_nodes, replace=False
            ).tolist()
            sparse_nodes = np.random.choice(
                self._sparse_nodes, num_sparse_nodes, replace=False
            ).tolist()
            choice = {
                "active_nodes": sorted(dense_nodes + sparse_nodes),
                "dense_in_dims": np.random.choice(self._dense_node_dims),
                "sparse_in_dims": np.random.choice(self._sparse_node_dims),
                'dense_sparse_interact': np.random.choice([0, 1]),
                "deep_fm": np.random.choice([0, 1]),
            }
            # Add constraint: dense/sparse node cannot be both zeros.
            if choice["active_nodes"] != self._zero_nodes:
                break
        return choice

    def _get_fixed_path_choice(self):
        """
        Fixed path sampling.
        Sample any of the nodes for the current choice blocks.
        Update: 07/20/2021: Use single-path strategy for supernet blocks.
        Different from '_get_any_path_choice' implemented next, this path sampling is NOT affected by variable 'anypath_choice'.
        This function should be called only once in 'fixed_path' strategy in runtime.
        """
        return self._get_single_path_choice()

    # Configure choice for sub-networks.
    def configure_choice(self, choice):
        self.choice = copy.deepcopy(choice)
        self.micro_last_choice = copy.deepcopy(choice)

    def discretize_config_each_block(
        self,
        probs,
        dense_nodes_topk=2,
        sparse_nodes_topk=1,
        in_dims_topk=2,
        include_zeros_3d=False,
    ):
        ops_config_cur_block = {
            "num_nodes": 0,
            "node_names": [],
            "dense_node_dims": [],
            "dense_nodes": [],
            "sparse_nodes": [],
            "zero_nodes": [],
        }
        node_cnt = 0
        # Process dense features.
        sorted_args = np.argsort(probs["dense_probs"])[::-1]
        sorted_args = sorted_args[:dense_nodes_topk]
        for node_idx in sorted_args:
            node_names_idx = self._dense_nodes[node_idx]
            ops_config_cur_block["node_names"].append(self._node_names[node_names_idx])
            ops_config_cur_block["dense_nodes"].append(node_cnt)
            if node_names_idx in self._zero_nodes:
                ops_config_cur_block["zero_nodes"].append(node_cnt)
            node_cnt += 1
        # Process sparse features.
        sorted_args = np.argsort(probs["sparse_probs"])[::-1]
        sorted_args = sorted_args[:sparse_nodes_topk]
        if include_zeros_3d:
            zeros_3d_idx = None
            for idx in range(len(self._sparse_nodes)):
                if self._node_names[self._sparse_nodes[idx]] == "zeros-3d":
                    zeros_3d_idx = idx
                    break
            assert (
                zeros_3d_idx is not None
            ), "'zeros-3d_idx' should not be None when 'include_zeros_3d' is True! Please check your config."
            if zeros_3d_idx not in sorted_args:
                sorted_args = np.append(sorted_args, zeros_3d_idx)

        for node_idx in sorted_args:
            node_names_idx = self._sparse_nodes[node_idx]
            ops_config_cur_block["node_names"].append(self._node_names[node_names_idx])
            ops_config_cur_block["sparse_nodes"].append(node_cnt)
            if node_names_idx in self._zero_nodes:
                ops_config_cur_block["zero_nodes"].append(node_cnt)
            node_cnt += 1
        ops_config_cur_block["num_nodes"] = node_cnt
        # Process in_dims
        sorted_args = np.argsort(probs["in_dims_probs"])[::-1]
        sorted_args = sorted_args[:in_dims_topk]
        for node_idx in sorted_args:
            node_names_idx = self._node_dims[node_idx]
            ops_config_cur_block["dense_node_dims"].append(self._node_dims[node_idx])
        ops_config_cur_block["dense_node_dims"] = sorted(ops_config_cur_block["dense_node_dims"])
        return ops_config_cur_block

    def to(self, *args):
        self = self.to(*args)
        self._device_args = args
