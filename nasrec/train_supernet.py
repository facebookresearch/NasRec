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
import os
import sys
sys.path.append(os.getcwd())
import warnings
warnings.simplefilter("ignore", ResourceWarning)
import argparse
# Important Imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# Project Imports
from nasrec.supernet.supernet import (
    SuperNet,
    ops_config_lib,
)
from nasrec.torchrec.utils import ParallelReadConcat
from nasrec.utils.config import (
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_CRITEO,
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_KDD,
)
from nasrec.utils.data_pipes import (
    get_criteo_kaggle_pipes,
    get_avazu_kaggle_pipes,
    get_kdd_kaggle_pipes,
)
from nasrec.utils.io_utils import (
    load_model_checkpoint,
    save_model_checkpoint,
    dump_pickle_data,
    create_dir,
)

# Project Imports
from nasrec.utils.train_utils import (
    get_l2_loss,
    train_and_test_one_epoch,
    get_model_flops_and_params,
    warmup_supernet_model,
    init_weights,
)
from nasrec.utils.lr_schedule import (
    CosineAnnealingWarmupRestarts,
    ConstantWithWarmup,
)


def train_and_eval_one_model(model, args):
    """Evaluate one certain model with the training pipeline and validation pipeline.
    Args:
        :params model: Backbone models used to train.
        :params args: Other hyperparameter settings.
    """
    get_pipes = {
        "criteo-kaggle": lambda: get_criteo_kaggle_pipes(args),
        "avazu": lambda: get_avazu_kaggle_pipes(args),
        'kdd': lambda: get_kdd_kaggle_pipes(args),
    }

    train_datapipes, test_datapipes, num_train_workers, num_test_workers = get_pipes[
        args.dataset
    ]()

    # Wrap up data-loader.5
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
        model = warmup_supernet_model(model, train_loader, args.gpu)

    flops, params = get_model_flops_and_params(model, train_loader, args.gpu)
    print("FLOPS: {:.4f} M \t Params: {:.4f} M".format(flops / 1e6, params / 1e6))

    model.configure_path_sampling_strategy(args.strategy)

    # Functional headers for training purposes.
    if args.loss_function == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            "Loss function {} is not implemented!".format(args.loss_function)
        )

    # L2 loss function.
    def _l2_loss_fn(model):
        """
        Customized Loss function. 
        Has an optional choice to discard embedding regularization and disabling bias decay.
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
    num_train_steps = (
        num_train_steps_per_epoch * args.num_epochs
    )
    num_warmup_steps = num_train_steps // 10
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

    if args.checkpoint_path is not None:
        checkpoint = load_model_checkpoint(args.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.apply(init_weights)

    # Load pretrained embeddings if any.
    print(model)

    # Create logging_dir.
    strategy_name = (
        args.strategy
        if args.strategy == "single-path"
        else args.strategy + "-" + args.anypath_choice
    )
    logging_dir = os.path.join(
        args.logging_dir,
        "supernet_{}blocks_layernorm{:d}_{}_lr{:.2f}_supernetwarmup_{}".format(
            args.num_blocks,
            args.use_layernorm,
            strategy_name,
            args.learning_rate,
            args.supernet_training_steps,
        ),
    )

    print("Logging in directory: {}".format(logging_dir))
    create_dir(logging_dir)
    
    # Initialize TensorBoard Writer.
    with torch.no_grad():
        int_x, cat_x, _ = next(iter(train_loader))
        int_x, cat_x = int_x.to(args.gpu)[0].unsqueeze_(0), cat_x.to(args.gpu)[0].unsqueeze_(0)
        writer = SummaryWriter(args.logging_dir)
        writer.add_graph(model, (int_x, cat_x))

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
                display_interval=args.display_interval,
                test_interval=args.test_interval,
                max_train_steps=num_train_steps_per_epoch,
                test_only_at_last_step=True,
                grad_clip_value=5.0,
                tb_writer=writer,
        )
        epoch_logs.append(logs)

    print("Dumping logs to {}!".format(logging_dir))
    dump_pickle_data(
        os.path.join(logging_dir, "train_test_logs.pickle"), logs
    )
    # Now, store the trained model weights.
    save_model_checkpoint(
        model, os.path.join(logging_dir, "supernet_checkpoint.pt"), optimizer
    )
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

# sparse-lr: 0.04, eps=1e-3
# dense-lr: 0.0002
# Opt: Adam (0.9, 0.999), eps=1e-8

def main(args):
    model = SuperNet(
        sparse_input_size=_num_sparse_inputs_dict[args.dataset],
        num_blocks=args.num_blocks,
        ops_config=ops_config_lib[args.config],
        use_layernorm=(args.use_layernorm == 1),
        activation="relu",
        num_embeddings=_num_embedding_dict[args.dataset],
        path_sampling_strategy=args.strategy,
        anypath_choice=args.anypath_choice,
        supernet_training_steps=args.supernet_training_steps,
        candidate_choices=None,
        # use_activation_at_residual_add=(args.enable_add_at_residual == 1),
    )
    # Copy model to GPU.
    model = model.to(args.gpu)
    train_and_eval_one_model(model, args)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

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
    # Strategies
    parser.add_argument(
        "--strategy", type=str, default="single-path", help="Path sampling strategy",
        choices=['evo-2shot-path', 'default', 'single-path', 'any-path', 'full-path', 'fixed-path']
    )
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
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path to load some pre-configured models.",
    )
    # Specific for 2nd shot only. Load the paths.
    parser.add_argument("--evo_2shot_path_candidates", type=str, default=None, \
        help="2nd shot path sampling candidates.")
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
        "--supernet_training_steps",
        type=int,
        default=2000,
        help="Supernet training steps. Supernet training probability will be linearly decayed to 0 when it hits 'supernet_training_steps'.",
    )
    parser.add_argument(
        "--anypath_choice",
        type=str,
        default="uniform",
        help="The way to perform any-path sampling.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=200, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16368, help="Testing batch size."
    )
    #-------------------Criteo------------------
    # Train: 36672495 Val: 4584061 Test: 4584061 Trainval: 41256556
    # ------------------Avazu-------------------
    # Train: 32343175 Val: 4042896 Test: 4042896 Trainval: 36386071
    # ------------------KDD---------------------
    # Train: 119711284 Val: 14963910 Test: 14963910 Trainval: 134675194
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
    parser.add_argument(
        "--lr_schedule", default="cosine", help="Learning rate schedule",
        choices=["cosine", "constant", "constant-no-warmup"])
    parser.add_argument(
        "--display_interval",
        type=int,
        default=100,
        help="Interval to display tensorboard curve/training stats.",
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=100,
        help="Testing interval when training supernet.",
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
        "--train_split",
        type=str,
        default="train",
        help="Data split for training. Can be one of ['train', 'trainval']",
        choices=["train", "trainval"],
    )
    parser.add_argument(
        "--validate_split",
        type=str,
        default="test",
        help="Data split for validation (evaluation). Can be one of ['val', test']",
        choices=["val", "test"],
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
        "--optimizer",
        type=str,
        default="adagrad",
        help="Optimizer to choose",
        choices=["adagrad", "sgd", "adam", "rmsprop"],
    )
    # Use embeddings during training.
    parser.add_argument("--pretrained_dlrm_emb_path", type=str, default=None,
        help="Pretrained embedding path from DLRM model.")
    # GPU utils
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    
    args = parser.parse_args()
    main(args)
