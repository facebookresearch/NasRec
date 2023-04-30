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
from itertools import count
import time
import copy

# Other imports
from typing import Union, Any, Optional
from math import sqrt

import numpy as np
from nasrec.supernet.modules import (
    DotProduct,
    ElasticLinear,
    SigmoidGating,
    Sum,
    Transformer,
)
import sklearn.metrics
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from nasrec.supernet.supernet import SuperNet
from fvcore.nn import FlopCountAnalysis


class DataLoaderIterator(object):
    """
    A improved dataloader which repeatedly iterates a given 'data_loader'.
    Batches with 'batch_size' will be produced, and 'drop_last_batch' can be enabled to
    remove smaller batches.
    """

    def __init__(self, data_loader, batch_size, drop_last_batch=True):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        # Create a dataloader iterator.
        self.data_loader_iterator = iter(self.data_loader)

    def get_next_batch(self):
        try:
            int_x_val, cat_x_val, y_val = next(self.data_loader_iterator)
            if y_val.size(0) < self.batch_size:
                raise StopIteration
        except StopIteration:
            self.data_loader_iterator = iter(self.data_loader)
            int_x_val, cat_x_val, y_val = next(self.data_loader_iterator)
        return int_x_val, cat_x_val, y_val


from math import sqrt
def init_weights(m):
    """
    Apply weight initialization to the module.
    Args:
        m (nn.Module): a PyTorch module.
    """
    if type(m) == nn.Embedding:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif type(m) == nn.MultiheadAttention:
        for p in m.parameters():
            if len(p.size()) > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    else:
        pass

def get_l2_loss(
    model: nn.Module,
    reg: float,
    no_reg_param_name: Union[str, None] = None,
    gpu: Optional[int] = None,
):
    """Compute L2 loss via a loss function.
    Args:
        :params reg (float): Weight decay.
        :params no_reg_param_name (bool): name of the layer that needs no regularization.
        :params gpu: GPU ID to use. 'None' defaults to CPU.
    """
    if reg == 0:
        return torch.tensor(0.0).to(gpu)
    reg_loss = None
    for n, m in model.named_parameters():
        # print(n, no_reg_param_name)
        # Bias is not regularized. Variables starting with 'no_reg_param_name' will not be regularized.
        if len(m.shape) == 1 or (
            (no_reg_param_name is not None) and n.startswith(no_reg_param_name)
        ):
            continue
        reg_loss_ = torch.square(torch.norm(m, p=2)) * reg
        reg_loss = reg_loss_ if reg_loss is None else reg_loss + reg_loss_
    return reg_loss


def accuracy(gt, pred):
    """
    Accuracy function.
    :params gt: GroundTruth Tensor.
    :params pred: Prediction Tensor (unscaled).
    """
    pred_binary = torch.gt(pred, 0.5).float()
    acc = (pred_binary == gt).sum() / pred_binary.size(0)
    return acc


def test_one_epoch(
    model, test_loader, loss_fn, gpu: Optional[int] = None, max_steps=-1, use_amp: bool = False,
):
    """Test the model for 1 epoch.
    Args:
        :params model: Backbone model.
        :params test_loader: Testing data loader.
        :params loss_fn: Loss function.
        :params gpu: GPU ID to use. 'None' defaults to CPU.
    """
    model.eval()
    counter = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for int_x, cat_x, y in test_loader:
            int_x, cat_x, y = int_x.to(gpu), cat_x.to(gpu), y.to(gpu)
            with torch.cuda.amp.autocast(enabled=use_amp):   
                pred = model(int_x, cat_x)
            y_pred.append(pred)
            y_true.append(y)
            counter += 1
            if counter % 50 == 0:
                print("done {} batches!".format(counter))
            if max_steps != -1 and counter >= max_steps:
                break

        print("Done {} batches!".format(counter))

        y_true_tensor = torch.cat(y_true).flatten()
        y_pred_tensor = torch.cat(y_pred).flatten() # Model does not provide sigmoid

        y_pred_tensor_sigmoid = torch.sigmoid(y_pred_tensor)


        y_pred_tensor_numpy_sigmoid = y_pred_tensor_sigmoid.detach().cpu().numpy()
        # y_pred_tensor_numpy = y_pred_tensor.detach().cpu().numpy()
        y_true_tensor_numpy = y_true_tensor.detach().cpu().numpy()

        # Whether use sigmoid or not does not change the result.
        auroc = sklearn.metrics.roc_auc_score(
            y_true_tensor_numpy,
            y_pred_tensor_numpy_sigmoid,
        )
        acc = accuracy(y_true_tensor, y_pred_tensor_sigmoid)
        test_loss = loss_fn(
            y_pred_tensor,
            y_true_tensor,
        )
    return acc.item(), auroc.item(), test_loss.item()


def train_and_test_one_epoch(
    model,
    epoch: int,
    optimizer: Any,
    lr_scheduler,
    train_loader,
    test_loader,
    loss_fn,
    l2_loss_fn,
    train_batch_size: int,
    gpu: Union[int, None],
    display_interval: int = 100,
    test_interval: int = 2000,
    max_train_steps: int = -1,
    max_eval_steps: int = -1,
    test_only_at_last_step: bool = False,
    grad_clip_value: float = None,
    tb_writer: Optional[SummaryWriter] = None,
    use_amp: bool = False,
):
    """Train & Test the model for 1 epoch on the training data pipeline.
    Testing process is inserted in-between.
    Args:
        :params model: Backbone model.
        :params epoch: Current epoch number.
        :params optimizer: Optimizer to optimize the model.
        :params train_loader: Training data loader.
        :params test_loader: Testing data loader.
        :params loss_fn: Loss function.
        :params l2_loss_fn: Loss function for L2 regularization.
        :params train_batch_size: Training batch size.
        :params gpu: GPU ID to use.
        :params display_interval: Interval of displaying training results.
        :params test_interval: Interval of carrying the testing to get test auroc.
        ;params max_train_steps: Number of maximum steps before termination.
        :params max_eval_steps: Number of maximum evaluation steps before termination.
        :params test_only_at_last_step: Only test at the last step.
        :params summary_writer_log_dir: The path to log tensorboard summary.
        :params grad_clip_value: Gradient clipping by norm.
        :params tb_writer: TensorBoard Summary Writer.
        :params use_amp: Whether use mixed precision (experimental).
    """
    # Get test batch size from test loader.
    _, _, y_val = next(iter(test_loader))
    test_batch_size = y_val.size(0)

    model.train()
    start_batch = time.time()
    end_batch = time.time()

    y_pred = []
    y_true = []

    logs = {
        "train_loss": [],
        "train_AUROC": [],
        "train_Accuracy": [],
        "test_loss": [],
        "test_AUROC": [],
        "test_Accuracy": [],
        "epoch": [],
        "iters": [],
    }

    start_gpu, end_gpu = 0.0, 0.0

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_model = None
    best_test_loss = 9999.99

    for batch_num, (int_x, cat_x, y) in enumerate(train_loader):
        end_batch = time.time()
        int_x = int_x.to(gpu, non_blocking=True)
        cat_x = cat_x.to(gpu, non_blocking=True)
        y = y.to(gpu, non_blocking=True)
        # Do a vanilla forward pass.
        start_gpu = time.time()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            res = model(int_x, cat_x)
            loss = loss_fn(res, y)
            l2_loss = l2_loss_fn(model)
            total_loss = loss + l2_loss

        # Drop the last batch which may contain insufficient examples.
        if len(y) == train_batch_size:
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                # Clip gradients.
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                # Update scale
                scaler.update()
                # optimizer.step()
            else:
                total_loss.backward()
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
        end_gpu = time.time()

        if batch_num % display_interval == 0 or batch_num == max_train_steps - 1:
            # First, log training loss.
            y_pred = res.detach()
            y_true = y.detach()

            # Check if nan. This happens in KDD.
            if torch.isnan(loss):
                print("Loss NaN. Exiting...")
                # The target model may diverge. Thus, return a series of large loss and low Accuracy/AUROC.
                logs["test_loss"].append(999.99)
                logs["test_AUROC"].append(-1)
                logs["test_Accuracy"].append(-1)
                return logs

            loss, l2_loss, current = loss.item(), l2_loss.item(), batch_num
            # This is optional. If you do not need to accurately profile GPU time, please just comment the next 2 lines.
            # torch.cuda.synchronize(gpu)
            print(f"Epoch: {epoch:>d} L2: {l2_loss:>7f} loss: {loss:>7f}  {current}")
            current_lr = lr_scheduler.get_lr()
            current_lr = current_lr[0] if isinstance(current_lr, list) else current_lr
            print("Learning rate: {}".format(current_lr))
            print(
                "Data: {:.5f} (s), GPU: {:.5f} (s)".format(
                    end_batch - start_batch, end_gpu - start_gpu
                )
            )

            # Also, print out AUROC for debugging.
            y_true_tensor = y_true.view(-1)
            y_pred_tensor = torch.sigmoid(y_pred).view(-1)  # Model does not provide sigmoid

            y_pred_tensor_numpy = y_pred_tensor.cpu().numpy()
            y_true_tensor_numpy = y_true_tensor.cpu().numpy()
            # Results are the same with/without sigmoid.
            try:
                train_auroc = sklearn.metrics.roc_auc_score(
                    y_true_tensor_numpy,
                    y_pred_tensor_numpy,
                )
            except Exception:
                print("AUROC encountered issues. All training data has the same label.")
                train_auroc = 1.0
            train_acc = accuracy(y_true_tensor, y_pred_tensor).item()
            print("Train Acc: {}, Train AUROC: {}".format(train_acc, train_auroc))

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train/epoch{}".format(epoch), loss, batch_num * train_batch_size)
                tb_writer.add_scalar("Acc/train/epoch{}".format(epoch), train_acc, batch_num * train_batch_size)
                tb_writer.add_scalar("AUROC/train/epoch{}".format(epoch), train_auroc, batch_num * train_batch_size)
                tb_writer.add_scalar("iters/epoch{}".format(epoch), batch_num * train_batch_size, batch_num * train_batch_size)

            # Logging for plot purposes.
            logs["train_loss"].append(loss)
            logs["train_AUROC"].append(train_auroc)
            logs["train_Accuracy"].append(train_acc)
            logs["epoch"].append(epoch)
            logs["iters"].append(batch_num)

        if batch_num % test_interval == 0 or batch_num == max_train_steps - 1:
            if (not test_only_at_last_step) or (
                test_only_at_last_step and batch_num == max_train_steps - 1
            ):
                # Now, going for testing.
                model.eval()
                test_start_time = time.time()
                test_acc, test_auroc, test_loss = test_one_epoch(
                    model, test_loader, loss_fn, gpu, max_steps=max_eval_steps, use_amp=use_amp
                )
                test_end_time = time.time()
                print(
                    "{:.4f} seconds elasped for testing!".format(
                        test_end_time - test_start_time
                    )
                )
                print(
                    "Test Acc: {}, Test AUROC: {}, Test Loss: {}".format(
                        test_acc, test_auroc, test_loss
                    )
                )
                logs["test_loss"].append(test_loss)
                logs["test_AUROC"].append(test_auroc)
                logs["test_Accuracy"].append(test_acc)

                if test_loss < best_test_loss:
                    best_model = copy.deepcopy(model)
                    best_test_loss = test_loss

                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/test/epoch{}".format(epoch), test_loss, batch_num  * train_batch_size)
                    tb_writer.add_scalar("Acc/test/epoch{}".format(epoch), test_acc, batch_num  * train_batch_size)
                    tb_writer.add_scalar("AUROC/test/epoch{}".format(epoch), test_auroc, batch_num  * train_batch_size)
                    tb_writer.add_scalar("Best Loss/test/epoch{}".format(epoch), best_test_loss, batch_num  * train_batch_size)

            model.train()
        # Added early termination.
        if max_train_steps != -1 and batch_num >= max_train_steps - 1:
            return logs
        lr_scheduler.step()
        start_batch = time.time()
    print("Batch counter total: {}".format(batch_num))
    model = copy.deepcopy(best_model)
    return logs

def warmup_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    gpu: Union[int, None],
):
    """
    Warmup the model and initialize all lazy layers.
    Args:
        :params model: PyTorch backbone model.
        :params train_loader: Training dataloader to provide training examples.
        :params gpu: GPU to place the model.
    """
    model = model.to(gpu)
    int_x, cat_x, _ = next(iter(train_loader))
    int_x = int_x.to(gpu)
    cat_x = cat_x.to(gpu)
    # Set to full path sampling for initialization first, and then proceed to single-path sampling.
    model(int_x, cat_x)
    return model


def warmup_supernet_model(
    model: nn.Module, train_loader: torch.utils.data.DataLoader, gpu
):
    """
    Warmup the model and initialize all lazy layers. Note that the model must be a 'SuperNet' instance.
    Args:
        :params model: PyTorch backbone model.
        :params train_loader: Training dataloader to provide training examples.
        :params gpu: GPU to place the model.
    """
    assert isinstance(model, SuperNet), NotImplementedError(
        "For 'warmup_supernet_model', the passed in model must be a 'SuperNet' object."
    )
    model = model.to(gpu)
    int_x, cat_x, y = next(iter(train_loader))
    int_x = int_x.to(gpu)
    cat_x = cat_x.to(gpu)
    # Set to full path sampling for initialization first, and then proceed to single-path sampling.
    model.configure_path_sampling_strategy("full-path")
    model(int_x, cat_x)
    return model


def get_model_flops_and_params(model, train_loader, gpu):
    """
    Get the flops and the number of params for a single model with batch size `1`.
    Args:
        :params model: PyTorch backbone model.
        :params train_loader: Training dataloader to provide training examples.
        :params gpu: GPU to place the model.
    """
    model = model.to(gpu)
    int_x, cat_x, y = next(iter(train_loader))
    int_x = int_x.to(gpu)
    cat_x = cat_x.to(gpu)
    # FlopCountAnalysis has limited support of operators. e.g. it does not support dot product
    flops = FlopCountAnalysis(model, (int_x, cat_x))
    params = sum(p.numel() for p in model.parameters())
    # Need to divide flops by batch size.
    return flops.total() / int_x.size(0), params


def get_model_latency(
    model, inputs, gpu, num_warmup_steps: int = 10, num_trials: int = 200
):
    """
    Get the latency for a single model.
    Args:
        :params model: PyTorch backbone model.
        :params train_loader: Training dataloader to provide training examples.
        :params gpu: GPU to place the model.
    """
    # Note: these 2 numbers 'num_warmup_steps' and 'num_trials' need to be investigated to save computing time.
    model = model.to(gpu)
    is_training = model.training
    model = model.eval()
    assert (
        len(inputs) == 2
    ), "Number of inputs should have exactly 2 items for RM models!"
    int_x, cat_x = inputs
    int_x = int_x.to(gpu)
    cat_x = cat_x.to(gpu)
    all_latency = []
    with torch.no_grad():
        for idx in range(num_warmup_steps + num_trials):
            start = time.time()
            _ = model(int_x, cat_x)
            torch.cuda.synchronize(gpu)
            end = time.time()
            if idx < num_warmup_steps:
                continue
            all_latency.append(end - start)

    if is_training:
        model = model.train()

    all_latency = np.asarray(all_latency)

    lat_high_thresh, lat_low_thresh = np.percentile(all_latency, 95), np.percentile(
        all_latency, 5
    )
    valid_indices = (all_latency >= lat_low_thresh) & (all_latency <= lat_high_thresh)
    all_latency = all_latency[valid_indices]

    mean_latency = np.mean(all_latency)
    std_latency = np.std(all_latency)
    return mean_latency, std_latency
