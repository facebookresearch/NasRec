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
import glob

# Other imports
import os

import torch

import nasrec.torchrec.criteo as criteo
import nasrec.torchrec.avazu as avazu
import nasrec.torchrec.kdd as kdd

from nasrec.utils.config import (
    NUM_EMBEDDINGS_CRITEO,
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_KDD,
)


def get_criteo_kaggle_pipes(args):
    assert args.validate_split in ["val", "test"], ValueError(
        "Invalid validation split! Should be in ['val', 'test']."
    )

    def get_shard_dirs(root_dir):
        return sorted(glob.glob(os.path.join(root_dir, "shard-*")))

    all_shard_dirs = get_shard_dirs(args.root_dir)
    num_train_workers = len(all_shard_dirs)
    num_test_workers = len(all_shard_dirs)
    print("Training directory...", all_shard_dirs)
    train_file = "train.txt" if args.train_split == "train" else "trainval.txt"
    train_datapipes = [
        criteo.criteo_kaggle(os.path.join(all_shard_dirs[idx], train_file))
            .batch(args.train_batch_size)
            .collate()
            .map(VanillaTransformCriteo)
        for idx in range(num_train_workers)
    ]
    # Testing dataset preferes larger chuncks due to larger batch sizes
    test_datapipes = [
        criteo.criteo_kaggle(
            os.path.join(all_shard_dirs[idx], "{}.txt".format(args.validate_split))
        )
            .batch(args.test_batch_size)
            .collate()
            .map(VanillaTransformCriteo)
        for idx in range(num_test_workers)
    ]
    return train_datapipes, test_datapipes, num_train_workers, num_test_workers


def get_avazu_kaggle_pipes(args):
    assert args.validate_split in ["val", "test"], ValueError(
        "Invalid validation split! Should be in ['val', 'test']."
    )

    def get_shard_dirs(root_dir):
        return sorted(glob.glob(os.path.join(root_dir, "shard-*")))

    all_shard_dirs = get_shard_dirs(args.root_dir)
    num_train_workers = len(all_shard_dirs)
    num_test_workers = len(all_shard_dirs)
    print("Training directory...", all_shard_dirs)
    train_file = "train.txt" if args.train_split == "train" else "trainval.txt"
    train_datapipes = [
        avazu.avazu_kaggle(os.path.join(all_shard_dirs[idx], train_file))
            .batch(args.train_batch_size)
            .collate()
            .map(VanillaTransformAvazu)
        for idx in range(num_train_workers)
    ]
    # Testing dataset preferes larger chuncks due to larger batch sizes
    test_file = "{}.txt".format(args.validate_split)
    test_datapipes = [
        avazu.avazu_kaggle(
            os.path.join(all_shard_dirs[idx], test_file)
        )
            .batch(args.test_batch_size)
            .collate()
            .map(VanillaTransformAvazu)
        for idx in range(num_test_workers)
    ]
    return train_datapipes, test_datapipes, num_train_workers, num_test_workers

def get_kdd_kaggle_pipes(args):
    assert args.validate_split in ["val", "test"], ValueError(
        "Invalid validation split! Should be in ['val', 'test']."
    )

    def get_shard_dirs(root_dir):
        return sorted(glob.glob(os.path.join(root_dir, "shard-*")))

    all_shard_dirs = get_shard_dirs(args.root_dir)
    num_train_workers = len(all_shard_dirs)
    num_test_workers = len(all_shard_dirs)
    print("Training directory...", all_shard_dirs)
    train_file = "train.txt" if args.train_split == "train" else "trainval.txt"
    train_datapipes = [
        kdd.kdd_kaggle(os.path.join(all_shard_dirs[idx], train_file))
            .batch(args.train_batch_size)
            .collate()
            .map(VanillaTransformKDD)
        for idx in range(num_train_workers)
    ]
    # Testing dataset preferes larger chuncks due to larger batch sizes
    test_file = "{}.txt".format(args.validate_split)
    test_datapipes = [
        kdd.kdd_kaggle(
            os.path.join(all_shard_dirs[idx], test_file)
        )
            .batch(args.test_batch_size)
            .collate()
            .map(VanillaTransformKDD)
        for idx in range(num_test_workers)
    ]
    return train_datapipes, test_datapipes, num_train_workers, num_test_workers

criteo_col_transforms = {
    **{
        name: lambda x: torch.log(torch.maximum(torch.zeros_like(x), x) + 1)
        for name in criteo.DEFAULT_INT_NAMES
    },
    **{
        name: lambda x, num_embeddings: x.fmod(num_embeddings - 1) + 1
        for name in criteo.DEFAULT_CAT_NAMES
    },
}


def VanillaTransformCriteo(batch):
    # Warning: 'DEFAULT_INT_NAMES' may be a bit shaky if torchrec depencencies change.
    #for key in batch.keys():
    #    if key.startswith("int_"):
    #        print(key, batch[key][0])
    #exit(-1)
    int_x = torch.cat(
        [
            criteo_col_transforms[col_name](batch[col_name].clone().detach().unsqueeze(0).T)
            for col_name in criteo.DEFAULT_INT_NAMES
            if col_name in criteo_col_transforms
        ],
        dim=1,
    )
    cat_x = torch.cat(
        [
            criteo_col_transforms[col_name](
                torch.tensor([int(v, 16) if v else -1 for v in batch[col_name]])
                .unsqueeze(0)
                .T,
                NUM_EMBEDDINGS_CRITEO[int(col_name.split("_")[-1])],
            )
            for col_name in criteo.DEFAULT_CAT_NAMES
            if col_name in criteo_col_transforms
        ],
        dim=1,
    )
    y = batch[criteo.DEFAULT_LABEL_NAME].clone().detach().unsqueeze(1).float()
    return int_x, cat_x, y


# No dense feature for avazu.
avazu_col_transforms = {
    **{
        name: lambda x: torch.zeros_like(x).float()
        for name in avazu.DEFAULT_INT_NAMES
    },
    **{
        name: lambda x, num_embeddings: x.fmod(num_embeddings - 1) + 1
        for name in avazu.DEFAULT_CAT_NAMES
    },
}

def VanillaTransformAvazu(batch):
    # Warning: 'DEFAULT_INT_NAMES' may be a bit shaky if torchrec depencencies change.
    int_x = torch.cat(
        [
            avazu_col_transforms[col_name](batch[col_name].clone().detach().unsqueeze(0).T)
            for col_name in avazu.DEFAULT_INT_NAMES
            if col_name in avazu_col_transforms
        ],
        dim=1,
    )
    cat_x = torch.cat(
        [
            avazu_col_transforms[col_name](
                torch.tensor([int(v, 16) if v else -1 for v in batch[col_name]])
                .unsqueeze(0)
                .T,
                NUM_EMBEDDINGS_AVAZU[int(col_name.split("_")[-1])],
            )
            for col_name in avazu.DEFAULT_CAT_NAMES
            if col_name in avazu_col_transforms
        ],
        dim=1,
    )
    y = batch[avazu.DEFAULT_LABEL_NAME].clone().detach().unsqueeze(1).float()
    return int_x, cat_x, y


kdd_col_transforms = {
    **{
        name: lambda x: torch.log(torch.maximum(torch.zeros_like(x), x) + 1)
        for name in kdd.DEFAULT_INT_NAMES
    },
    **{
        name: lambda x, num_embeddings: x.fmod(num_embeddings - 1) + 1
        for name in kdd.DEFAULT_CAT_NAMES
    },
}

def VanillaTransformKDD(batch):
    # Warning: 'DEFAULT_INT_NAMES' may be a bit shaky if torchrec depencencies change.
    int_x = torch.cat(
        [
            kdd_col_transforms[col_name](batch[col_name].clone().detach().unsqueeze(0).T)
            for col_name in kdd.DEFAULT_INT_NAMES
            if col_name in kdd_col_transforms
        ],
        dim=1,
    )
    cat_x = torch.cat(
        [
            kdd_col_transforms[col_name](
                torch.tensor([int(v, 16) if v else -1 for v in batch[col_name]])
                .unsqueeze(0)
                .T,
                NUM_EMBEDDINGS_KDD[int(col_name.split("_")[-1])],
            )
            for col_name in kdd.DEFAULT_CAT_NAMES
            if col_name in kdd_col_transforms
        ],
        dim=1,
    )
    y = batch[kdd.DEFAULT_LABEL_NAME].clone().detach().unsqueeze(1).float()
    return int_x, cat_x, y
