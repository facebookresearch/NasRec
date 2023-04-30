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
from math import gamma
import os
import sys
sys.path.append(os.getcwd())

import warnings
# Disable warnings.
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ResourceWarning)

import argparse

# Other Imports
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')           # Run on cluster.
from torch.utils.data import DataLoader

# Project Imports
from nasrec.supernet.supernet import (
    SuperNet,
    ops_config_lib,
)
from nasrec.torchrec.utils import ParallelReadConcat
from nasrec.utils.config import (
    NUM_EMBEDDINGS_CRITEO,
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_KDD,
)
from nasrec.utils.data_pipes import (
    get_criteo_kaggle_pipes,
    get_avazu_kaggle_pipes,
    get_kdd_kaggle_pipes,
)
from nasrec.utils.train_utils import (
    get_l2_loss,
    train_and_test_one_epoch,
    warmup_supernet_model,
)
from nasrec.utils.lr_schedule import (
    CosineAnnealingWarmupRestarts,
    ConstantWithWarmup,
)

from nasrec.searcher.searcher import Searcher
from nasrec.utils.io_utils import (
    dump_pickle_data,
    create_dir,
    load_model_checkpoint,
    load_pickle_data,
)

def finetune_and_eval_one_model(model, args, checkpoint):
    """Fine-tune one certain model with the training pipeline and validation pipeline.
    Args:
        :params model: Backbone models used to fine-tune.
        :params args: Other hyperparameter settings.
        :params checkpoint: Checkpoint weights needed to initialize models.
    """
    get_pipes = {
        "criteo-kaggle": lambda: get_criteo_kaggle_pipes(args),
        'avazu': lambda: get_avazu_kaggle_pipes(args),
        'kdd': lambda: get_kdd_kaggle_pipes(args),
    }
    train_datapipes, test_datapipes, num_train_workers, num_test_workers = get_pipes[
        args.dataset
    ]()

    # Wrap up data-loader.
    train_loader = DataLoader(
        ParallelReadConcat(*train_datapipes),
        batch_size=None,
        num_workers=num_train_workers,
    )
    test_loader = DataLoader(
        ParallelReadConcat(*test_datapipes),
        batch_size=None,
        num_workers=num_test_workers,
    )

    # Now, warmup the model to initialize lazy layers.
    with torch.no_grad():
        model = warmup_supernet_model(model, train_loader, args.gpu)

    model.configure_path_sampling_strategy("fixed-path")

    # Functional headers for training purposes.
    if args.loss_function == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            "Loss function {} is not implemented!".format(args.loss_function)
        )

    # Fine-tuning FLAG.
    # 0.21-0.23s when fine-tuning the whole network on Tesla M40.
    # 0.05-0.06s / 512 batch with finetune-last-only on Tesla M40.
    if args.finetune_whole_supernet == 0:
        # print("Use Layernorm calibration ...")
        # model.set_mode_to_layernorm_calibrate()
        print("Finetune last only ...")
        model.set_mode_to_finelune_last_only()
    else:
        print("Finetuning the whole supernet.")

    # L2 loss function.
    def _l2_loss_fn(model):
        """
        Customized Loss function. Has an optional choice to discard embedding regularization and disabling bias decay.
        """
        return get_l2_loss(model, args.wd, args.no_reg_param_name, gpu=args.gpu)

    l2_loss_fn = _l2_loss_fn
    optimizer_lib = {
        "adagrad": torch.optim.Adagrad(
            model.parameters(), lr=args.learning_rate, eps=1e-2,
        ),
        "adam": torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-8),
        "sgd": torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, nesterov=True, momentum=0.9
        ),
    }
    optimizer = optimizer_lib[args.optimizer]

    # Scheduler
    num_train_steps = args.max_train_steps * args.num_epochs
    num_warmup_steps = args.max_train_steps // 10

    if args.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
            max_lr=args.learning_rate,
            min_lr=1e-8,
        )
    elif args.lr_schedule == "constant":
        lr_scheduler = ConstantWithWarmup(
            optimizer, num_warmup_steps=num_warmup_steps)
    elif args.lr_schedule == "stepwise":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_train_steps // 3, num_train_steps * 2 // 3], gamma=0.2
        )
    else:
        # Do nothing.
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_train_steps * 10], gamma=0.1
        )
        
    # Load checkpoints.
    if checkpoint is not None:
        # print(checkpoint["model_state_dict"].keys())
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        """     No Loading Optimizer dict. Will cause problems.
        if "optimizer_state_dict" in checkpoint:
            print("Loading optimizer dict...")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        """
    
    # Important. Step optimizer to initial steps after reading state_dict.
    lr_scheduler.step(epoch=-1)
    
    # Train the model for 1 epoch.
    logs = train_and_test_one_epoch(
        model,
        0,
        optimizer,
        lr_scheduler,
        train_loader,
        test_loader,
        loss_fn,
        l2_loss_fn,
        args.train_batch_size,
        args.gpu,
        max_train_steps=args.max_train_steps,
        max_eval_steps=args.max_eval_steps,
        test_interval=max(2, args.max_train_steps),
        test_only_at_last_step=(args.test_only_at_last_step) == 1,
        grad_clip_value=5.0,
    )  # Use max here to avoid 0 intervals.

    torch.cuda.empty_cache()
    
    return {
        "choice": model.choice,
        "test_acc": logs["test_Accuracy"],
        "test_auroc": logs["test_AUROC"],
        "test_loss": logs["test_loss"],
    }

_num_sparse_inputs_dict = {
    'criteo-kaggle': 26,
    'avazu': 23,
    'kdd': 10
}
_num_embedding_dict = {
    'criteo-kaggle': NUM_EMBEDDINGS_CRITEO,
    'avazu': NUM_EMBEDDINGS_AVAZU,
    "kdd": NUM_EMBEDDINGS_KDD,
}

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("Logging dir: {}".format(args.logging_dir))
    create_dir(args.logging_dir)
    # Initialize searcher.
    if args.method == "random":
        searcher = Searcher(finetune_and_eval_one_model, args)
        all_results = searcher.random_search_from_supernet(
            args.random_budget,
            num_parallel_workers=args.num_parallel_workers,
            beta=args.beta,
            target_latency=args.target_latency,
            latency_batch_size=args.latency_batch_size,
            sorted=(args.random_sort_results == 1),
            top_k=args.random_search_topk,
            criterion=args.criterion,
        )
    elif args.method == "regularized-ea":
        searcher = Searcher(finetune_and_eval_one_model, args)
        all_results = searcher.regularized_evolution_from_supernet(
            num_parallel_workers=args.num_parallel_workers,
            sample_size=args.sample_size,
            init_population=args.init_population,
            n_childs=args.n_childs,
            n_generations=args.n_generations,
            beta=args.beta,
            target_latency=args.target_latency,
            latency_batch_size=args.latency_batch_size,
            top_k=args.ea_top_k,
            criterion=args.criterion,
        )
    elif args.method == "cached":
        assert (
            args.choice_from_pickle_file is not None
        ), "'--choice_from_pickle_file' should not be None if you want to train from cached records!"
        all_choices = load_pickle_data(args.choice_from_pickle_file)
        num_subnets = len(all_choices)
        print(
            "Evaluating {} subnets from record file: {}".format(
                num_subnets, args.choice_from_pickle_file
            )
        )
        all_results = []
        checkpoint = load_model_checkpoint(args.ckpt_path)
        for idx in range(num_subnets):
            print("Evaluating {} of {} networks!".format(idx, num_subnets))
            print("GT performance: {:.5f}".format(all_choices[idx]["test_loss"]))
            print(all_choices[idx]["choice"])
            model = SuperNet(
                sparse_input_size=_num_sparse_inputs_dict[args.dataset],
                num_blocks=args.num_blocks,
                ops_config=ops_config_lib[args.config],
                use_layernorm=(args.use_layernorm == 1),
                activation="relu",
                num_embeddings=_num_embedding_dict[args.dataset],
                path_sampling_strategy="full-path",
            )
            model.configure_choice(all_choices[idx]["choice"])
            # Copy model to GPU.
            model = model.to(args.gpu)
            results = finetune_and_eval_one_model(model, args, checkpoint)
            all_results.append(results)
    else:
        raise NotImplementedError("Method {} not supported!".format(args.method))

    # Dump search results.
    dump_pickle_data(
        os.path.join(args.logging_dir, "results.pickle"), all_results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Supernet configuration
    parser.add_argument(
        "--num_blocks", type=int, default=7, help="Number of blocks in a supernet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xlarge",
        help="Configuration for the supernet search.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the checkpoint."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="Maximum steps to train. '-1' to train a whole epoch.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=-1,
        help="Maximum steps to evaluate. '-1' to evaluate a whole epoch.",
    )
    # Search methods
    parser.add_argument(
        "--method",
        type=str,
        default="cached",
        help="Search method.",
        choices=["random", "regularized-ea", "cached", "regularized-ea-pred", "random-pred"],
    )
    # General settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="criteo-kaggle",
        help="Choice of datasets",
        choices=["criteo-kaggle", "avazu", "kdd"],
    )
    parser.add_argument(
        "--root_dir", type=str, default="r"
    )
    parser.add_argument(
        "--logging_dir", type=str, default=None, help="Directory to put loggings."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Data split for training. Can be one of ['train', 'trainval']",
        choices=["train", "trainval"],
    )
    parser.add_argument(
        "--validate_split",
        type=str,
        default="val",
        help="Data split for validation (evaluation). Can be one of ['val', test']",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--choice_from_pickle_file",
        type=str,
        default=None,
        help="Whether load pre-sampled cached choices from a pickle file.",
    )
    parser.add_argument(
        "--use_layernorm",
        type=int,
        default=0,
        help="Whether use layernorm in the supernet or not.",
    )
    parser.add_argument(
        "--finetune_whole_supernet",
        type=int,
        default=0,
        help="Whether finetune the whole supernet or not.",
    )
    # Hyperparameters.
    parser.add_argument(
        "--wd", type=float, default=0, help="L2 Weight decay"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--sparse_lr", type=float, default=0.003, help="Sparse Embeddings Learning rate"
    )
    parser.add_argument(
        "--learning_rate_decay", type=float, default=0, help="Learning rate decay."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--lr_schedule", default="cosine", help="Learning rate schedule",
        choices=["cosine", "constant", "constant-no-warmup", "stepwise"])
    parser.add_argument(
        "--train_batch_size", type=int, default=200, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16368, help="Testing batch size."
    )
    # 39291957 for DLRM split, 36672133 for AutoCTR split (train).
    parser.add_argument(
        "--train_limit",
        type=int,
        default=36672495,
        help="Maximum number of training examples.",
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        default=6548659,
        help="Maximum number of testing examples.",
    )
    # Currently useful in SparseNN-V2.
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use in this work.",
        choices=["relu", "silu"],
    )
    parser.add_argument(
        "--no-reg-param-name",
        type=str,
        default=None,
        help="Name of the parameters that do not need to be regularized.",
    )
    # loss functions
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce",
        help="Loss function to perform the task.",
        choices=["bce"],
    )
    parser.add_argument(
        "--test_only_at_last_step",
        type=int,
        default=0,
        help="Whether only test the last step.",
    )
    # GPU utils
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU ID to use."
    )
    # Settings for number of parallel workers.
    parser.add_argument(
        "--num_parallel_workers",
        type=int,
        default=1,
        help="Number of parallel workers.",
    )
    # Settings for the random search.
    parser.add_argument(
        "--random_search_topk",
        type=int,
        default=5,
        help="Top-k archs to perserve as a result of random search.",
    )
    parser.add_argument(
        "--random_budget",
        type=int,
        default=5,
        help="Number of subnets to evaluate. Exclusive with 'choice_from_pickle_file'.",
    )
    parser.add_argument(
        "--random_sort_results",
        type=int,
        default=0,
        help="Whether sort random search results or not.",
    )
    # Settings for evolutionary search.
    parser.add_argument(
        "--n_childs",
        type=int,
        default=8,
        help="Number of child architectures per generation.",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=50,
        help="Number of generations in total."
    )
    parser.add_argument(
        "--init_population", type=int, default=8, help="Initial population."
    )
    parser.add_argument(
        "--ea_top_k",
        type=int,
        default=5,
        help="Top-k children to keep each generation.",
    )
    parser.add_argument(
        "--ea_predictor_path",
        type=str,
        default=None,
        help="Predictor path for EA."
    )
    parser.add_argument(
        "--ea_predictor_in_dims", type=int, default=266,
        help="Predictor input dimensions."
    )    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=6,
        help="Sample size from initial population.",
    )
    parser.add_argument("--optimizer", type=str, default='adagrad',
        help="Optimizer to choose.",
        choices=["adagrad", "sgd", "adam", "rmsprop"],
        )
    # Latency-aware arguments.
    parser.add_argument(
        "--criterion",
        type=str,
        default="test_loss",
        help="Criterion for search."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="TuNAS accuracy-latency trade-off objective."
    )
    parser.add_argument(
        "--target_latency",
        type=float,
        default=-1,
        help="Target latency. If -1, will measure DLRM latency online to get the calibration baseline.",
    )
    parser.add_argument(
        "--latency_batch_size",
        type=int,
        default=512,
        help="Batch size when measuring latency.",
    )
    args = parser.parse_args()
    main(args)
