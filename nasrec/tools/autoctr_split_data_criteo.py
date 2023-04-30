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

import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# This is a preprocessing script that does not need to strictly follow lint instructions.
def main(args):  # noqa
    INT_FEATS = 13
    CAT_FEATS = 26

    all_labels = []
    num_counts = 0
    all_lines = []
    with open(args.data_path, "r") as fp:
        while True:
            line = fp.readline()
            if line == "":
                break
            splitted_lines = line.split("\t")
            label = int(splitted_lines[0])
            all_labels.append(label)
            num_counts += 1
            if num_counts % 1000000 == 0:
                print("Processed {} examples!".format(num_counts))
            all_lines.append(splitted_lines[INT_FEATS+1:])

    # Uncomment to process embeddings. This step is NOT NECESSARY.
    # NOTE: You may need to spend a long time and memory in doing this.
    """
    all_emb_contents_len = []
    print("Copying lists...")
    all_lines = np.asarray(all_lines)
    assert all_lines.shape[-1] == CAT_FEATS
    print("Done copying lists.")
    print("Preparing embeddings...")
    for idx in tqdm(range(CAT_FEATS)):
        _, indices = np.unique(all_lines[:, idx], return_index=True)
        all_emb_contents_len.append(len(indices))
    
    print(all_emb_contents_len)
    """

    all_labels = np.asarray(all_labels)
    total_num_splits = (
        args.num_train_splits + args.num_val_splits + args.num_test_splits
    )
    splitter = StratifiedKFold(
        n_splits=total_num_splits, shuffle=True, random_state=2018
    ).split(np.zeros_like(all_labels), all_labels)

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, split in enumerate(splitter):
        if idx < args.num_train_splits:
            train_indices.append(split[-1])
        elif (
            idx >= args.num_train_splits
            and idx < args.num_train_splits + args.num_val_splits
        ):
            val_indices.append(split[-1])
        else:
            test_indices.append(split[-1])

    train_indices = np.sort(np.concatenate(train_indices, axis=0))
    val_indices = np.sort(np.concatenate(val_indices, axis=0))
    test_indices = np.sort(np.concatenate(test_indices, axis=0))

    total_train_items, total_val_items, total_test_items = (
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    print("Training examples: {}".format(total_train_items))
    print("Validation examples: {}".format(total_val_items))
    print("Testing examples: {}".format(total_test_items))

    # Now, split data into multiple shards
    num_train_shard_idx, num_val_shard_idx, num_test_shard_idx = -1, -1, -1

    train_split_idx, val_split_idx, test_split_idx = 0, 0, 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_files_per_shard = round(total_train_items / args.num_shards)
    num_val_files_per_shard = round(total_val_items / args.num_shards)
    num_test_files_per_shard = round(total_test_items / args.num_shards)

    train_fp, val_fp, test_fp = None, None, None

    train_counter, val_counter, test_counter = 0, 0, 0
    counter = 0
    with open(args.data_path, "r") as fp:
        # Skip 1st line.
        fp.readline()
        while True:
            line = fp.readline()
            if line == "":
                break
            if counter == train_indices[train_split_idx]:
                if train_fp is None or train_counter >= num_train_files_per_shard:
                    num_train_shard_idx += 1
                    if train_fp is not None:
                        train_fp.close()
                    shard_folder = os.path.join(
                        args.output_dir, "shard-{:d}".format(num_train_shard_idx)
                    )
                    if not os.path.exists(shard_folder):
                        os.makedirs(shard_folder)
                    file_name = os.path.join(shard_folder, "train.txt")
                    train_fp = open(file_name, "w")  # noqa
                    print("Open {}!".format(file_name))
                    train_counter = 0
                train_fp.write(line)
                train_counter += 1
                train_split_idx = min(train_split_idx + 1, total_train_items - 1)

            if counter == val_indices[val_split_idx]:
                if val_fp is None or val_counter >= num_val_files_per_shard:
                    num_val_shard_idx += 1
                    if val_fp is not None:
                        val_fp.close()
                    shard_folder = os.path.join(
                        args.output_dir, "shard-{}".format(num_val_shard_idx)
                    )
                    if not os.path.exists(shard_folder):
                        os.makedirs(shard_folder)
                    file_name = os.path.join(shard_folder, "val.txt")
                    val_fp = open(file_name, "w")  # noqa
                    print("Open {}!".format(file_name))
                    val_counter = 0
                val_fp.write(line)
                val_counter += 1
                val_split_idx = min(val_split_idx + 1, total_val_items - 1)

            if counter == test_indices[test_split_idx]:
                if test_fp is None or test_counter >= num_test_files_per_shard:
                    num_test_shard_idx += 1
                    if test_fp is not None:
                        test_fp.close()
                    shard_folder = os.path.join(
                        args.output_dir, "shard-{}".format(num_test_shard_idx)
                    )
                    if not os.path.exists(shard_folder):
                        os.makedirs(shard_folder)
                    file_name = os.path.join(shard_folder, "test.txt")
                    print("Open {}!".format(file_name))
                    test_fp = open(file_name, "w")  # noqa
                    test_counter = 0
                test_fp.write(line)
                test_counter += 1
                test_split_idx = min(test_split_idx + 1, total_test_items - 1)
            counter += 1
            if counter % 100000 == 0:
                print("Written {} examples!".format(counter))

        if train_fp is not None:
            train_fp.close()
        if val_fp is not None:
            val_fp.close()
        if test_fp is not None:
            test_fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Data path.")
    parser.add_argument(
        "--num_train_splits", type=int, default=8, help="Number of training splits."
    )
    parser.add_argument(
        "--num_val_splits", type=int, default=1, help="Number of validation splits."
    )
    parser.add_argument(
        "--num_test_splits", type=int, default=1, help="Number of testing splits."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for prepared files.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=8, help="Number of shards per dataset."
    )
    args = parser.parse_args()
    main(args)
