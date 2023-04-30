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

# Important Imports
# Other Imports
import os
import sys
sys.path.append(os.getcwd())

import argparse
import warnings
warnings.simplefilter("ignore", ResourceWarning)

import numpy as np
import torch

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

from nasrec.utils.lr_schedule import (
    CosineAnnealingWarmupRestarts,
    ConstantWithWarmup,
)

from nasrec.utils.train_utils import (
    get_l2_loss,
    train_and_test_one_epoch,
    warmup_model,
    get_model_flops_and_params,
    init_weights,
)
from torch.utils.data import DataLoader
# I/O utils.
from nasrec.utils.io_utils import (
    dump_pickle_data,
    load_pickle_data,
    create_dir,
)


def train_and_eval_one_model(model, args):
    """Evaluate one certain model with the training pipeline and validation pipeline.
    Args:
        :params model: Backbone models used to train.
        :params args: Other hyperparameter settings.
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

    with torch.no_grad():
        model = warmup_model(model, train_loader, args.gpu)
    flops, params = get_model_flops_and_params(model, train_loader, args.gpu)
    print("FLOPS: {:.4f} M \t Params: {:.4f} M".format(flops / 1e6, params / 1e6))
    # Functional headers for training purposes.
    if args.loss_function == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            "Loss function {} is not implemented!".format(args.loss_function)
        )

    model.apply(init_weights)

    # L2 loss function.
    def _l2_loss_fn(model):
        """
        Customized Loss function. Has an optional choice to discard embedding regularization and disabling bias decay.
        """
        return get_l2_loss(model, args.wd, args.no_reg_param_name, gpu=args.gpu)

    l2_loss_fn = _l2_loss_fn
    # Optimizer
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
    num_train_steps_per_epoch = args.train_limit // args.train_batch_size
    num_train_steps = num_train_steps_per_epoch * args.num_epochs
    num_warmup_steps = num_train_steps_per_epoch // 10 // args.num_epochs
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
    else:
        # Do nothing.
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_train_steps * 10], gamma=0.1
        )

    epoch_logs = []
    for epoch in range(args.num_epochs):
        # Train the model for 1 epoch.
        logs = train_and_test_one_epoch(
            model,
            epoch,
            optimizer,
            lr_scheduler,
            train_loader,
            test_loader,
            loss_fn,
            l2_loss_fn,
            args.train_batch_size,
            args.gpu,
            test_interval=args.test_interval,
            max_train_steps=num_train_steps_per_epoch
            if args.max_train_steps == -1
            else args.max_train_steps,
            max_eval_steps=args.max_eval_steps,
            test_only_at_last_step=True,
            grad_clip_value=5.0,
        )
        logs["flops(M)"], logs["Params(M)"] = flops / 1e6, params / 1e6
        epoch_logs.append(logs)
    return epoch_logs

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
    # Set random seed for sampling.
    np.random.seed(args.random_seed)
    if args.choice_from_pickle_file is not None:
        all_choices = load_pickle_data(args.choice_from_pickle_file)
        num_subnets = len(all_choices)
        print(
            "Evaluating {} subnets from record file: {}".format(
                num_subnets, args.choice_from_pickle_file
            )
        )
    else:
        num_subnets = args.num_subnets
        all_choices = None

    all_results = []
    # Create directory.
    create_dir(args.logging_dir)
    for i in range(num_subnets):
        choice = None if all_choices is None else all_choices[i]["choice"]
        print("Evaluating {:d} out of {:d} subnetworks!".format(i, num_subnets))
        model = SuperNet(
            sparse_input_size=_num_sparse_inputs_dict[args.dataset],
            num_blocks=args.num_blocks,
            ops_config=ops_config_lib[args.config],
            use_layernorm=(args.use_layernorm == 1),
            activation=args.activation,
            num_embeddings=_num_embedding_dict[args.dataset],
            path_sampling_strategy="fixed-path",
            fixed=True,
            fixed_choice=choice,
        )
        # Copy model to GPU.
        model = model.to(args.gpu)
        logs = train_and_eval_one_model(model, args)
        print("Trained model with the following choice...")
        print(model.choice)
        if logs[0]["test_AUROC"][-1] < 0:
            print("Model Diverged! Skip logging...")
            i -= 1
            continue
        results = {
            "choice": model.choice,
            "test_acc": logs[0]["test_Accuracy"][-1],
            "test_auroc": logs[0]["test_AUROC"][-1],
            "test_loss": logs[0]["test_loss"][-1],
        }
        all_results.append(results)

        # Save analysis files.
        dump_pickle_data(
            os.path.join(args.logging_dir, "results.pickle"), all_results
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="criteo-kaggle",
        help="Choice of datasets",
        choices=["criteo-kaggle", "avazu", "kdd"],
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="Root Directory for dataset."
    )
    parser.add_argument(
        "--logging_dir", type=str, default=None, help="Directory to put loggings."
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
    # supernet (subnet) configs.
    parser.add_argument(
        "--use_layernorm",
        type=int,
        default=0,
        help="Whether use layernorm in the supernet or not.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xlarge",
        help="Configuration for the supernet search.",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=7, help="Number of blocks per supernet."
    )
    parser.add_argument(
        "--random_seed", type=int, default=None, help="Random seed to carry sampling."
    )
    # Hyperparameters.
    parser.add_argument("--wd", type=float, default=0, help="L2 Weight decay")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--learning_rate_decay", type=float, default=0, help="Learning rate decay."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=200, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16368, help="Testing batch size."
    )
    parser.add_argument(
        "--lr_schedule", default="cosine", help="Learning rate schedule",
        choices=["cosine", "constant", "constant-no-warmup"])
    # 36672495 for AutoCTR split (train).
    parser.add_argument(
        "--train_limit",
        type=int,
        default=36672495,
        help="Maximum number of training examples.",
    )
    # 4296061 for AutoCTR split (val, test),
    parser.add_argument(
        "--test_limit",
        type=int,
        default=4296061,
        help="Maximum number of testing examples.",
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=100,
        help="Testing interval when training supernet.",
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
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use in this work.",
        choices=["relu", "silu"],
    )
    parser.add_argument(
        "--num_subnets", type=int, default=20, help="Number of experimental subnets."
    )
    parser.add_argument(
        "--choice_from_pickle_file",
        type=str,
        default=None,
        help="Whether load pre-sampled cached choices from a pickle file. If the file is provided, \
        the total number of records will override 'num_subnets' during subnet evaluation.",
    )
    parser.add_argument(
        "--no-reg-param-name",
        type=str,
        default=None,
        help="Name of the parameters that do not need to be regularized.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adagrad",
        help="Optimizer",
        choices=["adagrad", "sgd", "adam", "rmsprop"],
    )
    # loss functions
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce",
        help="Loss function to perform the task.",
        choices=["bce"],
    )
    # GPU utils
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    args = parser.parse_args()
    main(args)
