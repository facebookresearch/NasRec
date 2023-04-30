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

# Other imports
import numpy as np
import torch
# Project Imports
from nasrec.supernet.supernet import (
    SuperNet,
    ops_config_lib,
)
from nasrec.torchrec.criteo import (
    INT_FEATURE_COUNT,
    CAT_FEATURE_COUNT,
)
from nasrec.utils.config import (
    NUM_EMBEDDINGS_CRITEO,
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_KDD,
)
from nasrec.utils.io_utils import (
    load_model_checkpoint
)
from nasrec.utils.train_utils import get_model_latency


def get_device_id(job_id, on_cpu=False):
    return None if on_cpu else job_id


_num_embedding_dict = {
    'criteo-kaggle': NUM_EMBEDDINGS_CRITEO,
    'avazu': NUM_EMBEDDINGS_AVAZU,
    'kdd': NUM_EMBEDDINGS_KDD,
}
_num_sparse_inputs_dict = {
    'criteo-kaggle': 26,
    'avazu': 23,
    'kdd': 10
}

# Eval and fetch results for a single network.
# Note: this function is usually executed under a multi-processing mode!
def _create_model_train_and_get_results(
    args, gpu_id, eval_fn, tokenizer, choice, checkpoint, kwargs
):
    args.gpu = gpu_id
    model = SuperNet(
        sparse_input_size=_num_sparse_inputs_dict[args.dataset],
        num_blocks=args.num_blocks,
        ops_config=ops_config_lib[args.config],
        use_layernorm=(args.use_layernorm == 1),
        activation="relu",
        num_embeddings=_num_embedding_dict[args.dataset],
        path_sampling_strategy="full-path",
    )
    if choice is not None:
        model.configure_choice(choice)
    results = eval_fn(model, args, checkpoint)
    token = tokenizer.tokenize(model.choice)
    hash_token = tokenizer.hash_token(token)
    results["hash_token"] = hash_token
    # Get the latency of the model.
    if "beta" in kwargs and kwargs["beta"] != 0.00:
        cur_choice = model.choice
        del model
        int_x = torch.rand(
            (kwargs["latency_batch_size"], INT_FEATURE_COUNT), dtype=torch.float32
        )
        cat_x = torch.randint(
            0,
            1,
            size=(kwargs["latency_batch_size"], CAT_FEATURE_COUNT),
            dtype=torch.int32,
        )
        # Update: model has to be a standalone model so that latency can be measured.
        print("Getting latency of fixed model...")
        model_fixed = SuperNet(
            num_blocks=args.num_blocks,
            ops_config=ops_config_lib[args.config],
            use_layernorm=(args.use_layernorm == 1),
            activation="relu",
            num_embeddings=_num_embedding_dict[args.dataset],
            path_sampling_strategy="fixed-path",
            fixed=True,
            fixed_choice=cur_choice,
        )
        mean_lat, _ = get_model_latency(model_fixed, (int_x, cat_x), gpu_id)
        results["latency"] = mean_lat
        print("Latency: {:.5f} s.".format(mean_lat))
    return results


# Define shared resource bucket.
# Note: this function is usually executed under a multi-processing mode!
def create_model_train_and_get_results_helper(
    args, gpu_id, eval_fn, tokenizer, choice, return_dict, ckpt_holder, kwargs
):
    """
    A helper that is utilized to realize multi-processing for multiple search algorithms (for example, regularized evolution).
    """
    # Use the current time as seed.
    if "ckpt" not in list(ckpt_holder.keys()):
        if args.ckpt_path is not None:
            ckpt_holder["ckpt"] = load_model_checkpoint(args.ckpt_path)
        else:
            ckpt_holder["ckpt"] = None

    checkpoint = ckpt_holder["ckpt"]
    np.random.seed(None)
    return_dict["worker_{}".format(gpu_id)] = _create_model_train_and_get_results(
        args, gpu_id, eval_fn, tokenizer, choice, checkpoint, kwargs=kwargs
    )

