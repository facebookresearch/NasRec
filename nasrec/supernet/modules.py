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

# Important imports.
# Modules:
from math import sqrt
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_MHA_HEADS = 8

_activation_fn_lib = {
    "relu": lambda x: torch.nn.functional.relu(x, inplace=True),
    "silu": lambda x: torch.nn.functional.silu(x, inplace=True),
    "identity": lambda x: x,
}


def apply_activation_fn(x, activation):
    return _activation_fn_lib[activation](x)


# DEBUG FLAG. 'DEBUG=False' to reduce memory in training.
# 'DEBUG=True' allows visualizing the graph in tensorboard. Please set to 'False' for production.
class FLAGS:
    """
    General FLAGs for configuring the supernet.
    'DEBUG' (defaults to False): whether try some debugging features, for example, visualizing network in tensorboard.
    """

    def __init__(self):
        self.DEBUG = False

    def config_debug(self, debug: bool = False):
        self.DEBUG = debug


flags = FLAGS()


class CleverMaskGenerator:
    """
    This is a clever mask generator that will help in supernet training efficiency.
    # All generated masks will be cached and reused if needed.
    """

    def __init__(self):
        self.cached_mask = {}

    def __call__(
        self,
        max_dims_or_dims: int,
        dims_in_use: int,
        device: Optional[Union[int, torch.device]] = None,
    ):
        """
        Args:
            :params max_dims_or_dims (int): Maximum number of dimension in the generated mask,
             or dimension for a fixed subnet.
            :params dims_in_use (int): Dimension in use.
            :params device (int or None): Name of device to place this mask. 'None' places to CPU.
        """
        assert (
            max_dims_or_dims >= dims_in_use
        ), "'max_dims_or_dims' should be larger than 'dims_in_use' to successfully generate a mask."
        token = "{}_{}".format(max_dims_or_dims, dims_in_use)
        if token in self.cached_mask.keys() and not flags.DEBUG:
            return self.cached_mask[token]
        else:
            """
            if not flags.DEBUG:
                print("Cache miss for token {} in masks!".format(token))
            """
            mask = torch.cat(
                [torch.ones(dims_in_use), torch.zeros(max_dims_or_dims - dims_in_use)],
                dim=-1,
            ).to(device)
            mask.requires_grad = False
            self.cached_mask[token] = mask
            return mask


class CleverZeroTensorGenerator:
    """
    This is a clever zeros generator that will help in supernet training efficiency.
    All generated torch.zeros will be cached and reused if needed.
    """

    def __init__(self):
        self.cached_zeros = {}

    def __call__(
        self, size: torch.Size, device: Optional[Union[int, torch.device]] = None
    ):
        """
        Args:
            :params size (torch.Size): Shape of the zero tensor.
            :params device (int or None): Name of device to place this mask. 'None' places to CPU.
        """
        token = "_".join([str(x) for x in size])
        if token in self.cached_zeros and not flags.DEBUG:
            return self.cached_zeros[token]
        else:
            """
            if not flags.DEBUG:
                print("Cache miss for token {} in zeros!".format(token))
            """
            zeros = torch.zeros(size, dtype=torch.float).to(device)
            zeros.requires_grad = False
            self.cached_zeros[token] = zeros
            return zeros


_mask_generator = CleverMaskGenerator()
_zeros_generator = CleverZeroTensorGenerator()


class ElasticLinear(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Initialize a ElasticLinear class for supernet building.
        Args:
        :params fixed (bool): Whether fix this linear layer or not. If 'fixed' is True,
        masking will not be utilized to create sub-networks.
        Potential Kwargs:
        :params use_layernorm (bool): Whether attach layernorm at the end of elastic linear.
        :params max_dims_or_dims (int): The maximum dimension to project the output tensor from elastic linear,
        or dimension for a fixed subnet.
        :params activation (str): Activation function.
        """
        super(ElasticLinear, self).__init__()
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._activation = kwargs["activation"]
        self._activation_fn = _activation_fn_lib[self._activation]
        self._use_layernorm = kwargs["use_layernorm"]
        self._fixed = fixed
        # Main module: linear layers.
        self._linear = nn.LazyLinear(
            self._max_dims_or_dims, bias=not self._use_layernorm
        )
        if self._use_layernorm:
            self._layernorm = nn.LayerNorm([self._max_dims_or_dims])
        else:
            self._layernorm = None

    def forward(self, tensor, dims_in_use):
        if not self._fixed:
            assert dims_in_use <= self._max_dims_or_dims, ValueError(
                "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                    dims_in_use, self._max_dims_or_dims
                )
            )
        # Go through a vanilla linear.
        out = self._linear(tensor)
        # Apply layer norm if needed.
        if self._layernorm is not None:
            out = self._layernorm(out)
        # Generate & Apply the mask.
        if not self._fixed:
            mask = _mask_generator(self._max_dims_or_dims, dims_in_use, tensor.device)
            out = torch.multiply(self._activation_fn(out), mask)
        else:
            out = self._activation_fn(out)
        return out


class ElasticLinear3D(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Initialize a ElasticLinear class for supernet building.
        Args:
        :params fixed (bool): Whether fix this linear layer or not. If 'fixed' is True,
        masking will not be utilized to create sub-networks.
        Potential Kwargs:
        :params use_layernorm (bool): Whether attach layernorm at the end of elastic linear.
        :params max_dims_or_dims (int): The maximum dimension to project the output tensor from elastic linear,
        or dimension for a fixed subnet.
        :params activation (str): Activation function.
        """
        super(ElasticLinear3D, self).__init__()
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._activation = kwargs["activation"]
        self._activation_fn = _activation_fn_lib[self._activation]
        self._use_layernorm = kwargs["use_layernorm"]
        self._fixed = fixed
        # Main module: linear layers.
        self._linear = nn.LazyLinear(
            self._max_dims_or_dims, bias=not self._use_layernorm
        )
        if self._use_layernorm:
            self._layernorm = nn.LayerNorm([self._max_dims_or_dims])
        else:
            self._layernorm = None

    def forward(self, tensor, dims_in_use):
        assert len(tensor.size()) == 3, "Tensor should be 3D!"
        if not self._fixed:
            assert dims_in_use <= self._max_dims_or_dims, ValueError(
                "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                    dims_in_use, self._max_dims_or_dims
                )
            )
        # Go through a vanilla linear.
        out = tensor.transpose(1, 2)
        out = self._linear(out)
        # Apply layer norm if needed.
        if self._layernorm is not None:
            out = self._layernorm(out)
        # Generate & Apply the mask.
        if not self._fixed:
            mask = _mask_generator(self._max_dims_or_dims, dims_in_use, tensor.device)
            out = torch.multiply(self._activation_fn(out), mask)
        else:
            out = self._activation_fn(out)
        # Transpose the tensor back.
        out = out.transpose(1, 2)
        return out


class Zeros2D(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Self attention for 3D inputs. Basically, takes in (Q,K,V) as 3D tensors and output a 3D tensor.
        Args:
            :params fixed (bool): Whether fixing this layer or not. If fixed, no masking will be utilized when
            creating sub-networks with this block.
        Potential Kwargs:
            :params max_dims_or_dims: Maximum dimension or dimension of the zero tensor.
        """
        super(Zeros2D, self).__init__()
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._fixed = fixed

    def forward(self, dense_t: torch.Tensor, dims_in_use: int):
        assert len(dense_t.size()) == 2, ValueError(
            "Input tensor to 'Zeros2D' should have a 2D shape."
        )
        if not self._fixed:
            assert dims_in_use <= self._max_dims_or_dims, ValueError(
                "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                    dims_in_use, self._max_dims_or_dims
                )
            )
        if not self._fixed:
            return _zeros_generator(
                torch.Size((dense_t.size(0), self._max_dims_or_dims)), dense_t.device
            )
        else:
            return _zeros_generator(
                torch.Size((dense_t.size(0), dims_in_use)), dense_t.device
            )


class DotProduct(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Dot product class. This class takes in both sparse tensors and dense tensors to produce an interaction.
        Args:
        :params fixed (bool): Whether fix this dot-product layer or not. If 'fixed' is True,
        Potential Kwargs:
        :params use_layernorm (bool): Whether attach layernorm at the end of dot product.
        :params max_dims_or_dims (int): The maximum dimension to project the output tensor from dot-product,
        or dimension for a fixed subnet.
        :params embedding_dims (int): The embedding dim. Usually used to project dense/sparse tensors before proceeding.
        """
        super(DotProduct, self).__init__()
        self._use_layernorm = kwargs["use_layernorm"]
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._embedding_dim = kwargs["embedding_dim"]
        self._fixed = fixed
        # Intermediate linear projection layers to project dense inputs.
        self._dense_proj = nn.LazyLinear(
            self._embedding_dim, bias=not self._use_layernorm
        )
        self._sparse_proj = nn.LazyLinear(
            self._embedding_dim, bias=not self._use_layernorm
        )
        # Project sparse in 'num_inputs' dim to save parameters.
        self.sparse_inp_proj_dim = round(sqrt(2 * self._max_dims_or_dims))
        self._sparse_inp_proj = nn.LazyLinear(
            self.sparse_inp_proj_dim, bias=not self._use_layernorm,
        )
        # Final linear projection layers.
        self._linear_proj = nn.LazyLinear(
            self._max_dims_or_dims, bias=not self._use_layernorm
        )
        # Follow DLRM. Here, we change the module in forward function as we want to eliminate redundant modules that are not needed when
        # initializing standalone subnets. This is very useful and completed when the model goes through the warmup stage.
        self._dense_layernorm = (
            nn.LayerNorm(self._embedding_dim) if self._use_layernorm else None
        )
        self._sparse_layernorm = (
            nn.LayerNorm(self._embedding_dim) if self._use_layernorm else None
        )
        self._sparse_inp_proj_layernorm = (
            nn.LayerNorm(self.sparse_inp_proj_dim) if self._use_layernorm else None
        )
        self._linear_layernorm = (
            nn.LayerNorm(self._max_dims_or_dims) if self._use_layernorm else None
        )

    def forward(self, dense_t: torch.Tensor, sparse_t: torch.Tensor, dims_in_use: int):
        """
        Interact both dense features 'dense_t' and sparse features 'sparse_t'.
        """
        assert len(dense_t.size()) == 2, ValueError(
            "Dense tensor should be 2D, but found size {}!".format(dense_t.size())
        )
        assert len(sparse_t.size()) == 3, ValueError(
            "Sparse tensor should be 3D, but found size {}!".format(sparse_t.size())
        )
        if not self._fixed:
            assert dims_in_use <= self._max_dims_or_dims, ValueError(
                "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                    dims_in_use, self._max_dims_or_dims
                )
            )
        # Check dense_t shape and maybe linear projection to 'embedding_dim'.
        if dense_t.size(-1) != self._embedding_dim:
            x = self._dense_proj(dense_t)
            x = self._dense_layernorm(x) if self._use_layernorm else x
        else:
            x = dense_t
            self._dense_proj = None
            self._dense_layernorm = None

        # Check sparse_t shape and maybe linear projection to 'embedding_dim'.
        if sparse_t.size(-1) != self._embedding_dim:
            y = self._sparse_proj(sparse_t)
            y = self._sparse_layernorm(y) if self._use_layernorm else y
        else:
            y = sparse_t
            self._sparse_proj = None
            self._sparse_layernorm = None

        # Project sparse tensor in terms of 'num_inputs'.
        if y.size(1) != self.sparse_inp_proj_dim:
            y = y.transpose(1, 2)
            y = self._sparse_inp_proj(y)
            y = self._sparse_inp_proj_layernorm(y) if self._use_layernorm else y
            y = y.transpose(1, 2)
        else:
            self._sparse_inp_proj = None
            self._sparse_inp_proj_layernorm = None

        (batch_size, d) = x.shape
        # Now, carry the dot product: concatenate dense and sparse features
        T = torch.cat([torch.unsqueeze(x, 1), y], dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        _, ni, nj = Z.shape
        assert ni == nj, "dot product should produce a square matrix"
        # Improvement: use tril_indices to improve speed.
        offset = -1
        # 01/17/2022: Enable self-interaction.
        # offset = 0
        li, lj = torch.tril_indices(ni, nj, offset=offset)
        Zflat = Z[:, li, lj]
        assert len(Zflat.shape) == 2, "dot product should be 2D"
        # concatenate dense features and interactions
        # R = torch.cat([dense_t] + [Zflat], dim=1)
        R = Zflat
        if R.size(-1) != self._max_dims_or_dims:
            out = self._linear_proj(R)
        else:
            # Reset useless layers to None and reduce model size.
            out = R
            self._linear_proj = None

        # We should not skip layernorm here as DotProduct does not necessarily produce normalized tensors.
        out = self._linear_layernorm(out) if self._use_layernorm else out
        # Mask both tensors as they have limited dimensions.
        mask = (
            _mask_generator(self._max_dims_or_dims, dims_in_use, x.device)
            if not self._fixed
            else None
        )

        out = torch.multiply(out, mask) if not self._fixed else out
        return out

def _pad_2Dtensors_if_needed(
    left_2d_tensor: torch.Tensor, right_2d_tensor: torch.Tensor
):
    """
    This method aligns the dimension of the tensors if needed.
    """
    assert len(left_2d_tensor.size()) == 2, ValueError(
        "'left_2d_tensor' should have a 2D shape, but had shape: {}".format(
            left_2d_tensor.size()
        )
    )
    assert len(right_2d_tensor.size()) == 2, ValueError(
        "'right_2d_tensor' should have a 2D shape, but had shape: {}".format(
            right_2d_tensor.size()
        )
    )
    size_left = left_2d_tensor.size(-1)
    size_right = right_2d_tensor.size(-1)
    if size_left == size_right:
        return left_2d_tensor, right_2d_tensor
    padded_zeros = _zeros_generator(
        torch.Size((left_2d_tensor.size(0), abs(size_left - size_right))),
        device=left_2d_tensor.device,
    )
    if size_left < size_right:
        return torch.cat([left_2d_tensor, padded_zeros], dim=1), right_2d_tensor
    else:
        return left_2d_tensor, torch.cat([right_2d_tensor, padded_zeros], dim=1)

class Sum(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Sum 2 2-D Dense Tensors.
        Args:
            :params fixed (bool): whether fix this sum layer or not. If fixed, no masking will be used when creating
            sub-networks with this block.
        Potential Kwargs:
            :params use_layernorm (bool): Whether attach layernorm at the end of sum.
            :params max_dims_or_dims (int): The maximum dimension to project the output tensor from sum,
            or dimension for a fixed subnet.
        """
        super(Sum, self).__init__()
        self._use_layernorm = kwargs["use_layernorm"]
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._linear_proj = nn.LazyLinear(self._max_dims_or_dims, bias=not self._use_layernorm)
        # layernorm is still needed here, as the summed tensor after linear projection may not be normalized.
        if self._use_layernorm:
            self._layernorm = nn.LayerNorm(self._max_dims_or_dims)
        else:
            self._layernorm = None
            # self._left_ln = None
            # self._right_ln = None
        self._fixed = fixed


    def forward(self, left_2d: torch.Tensor, right_2d: torch.Tensor, dims_in_use: int):
        assert len(left_2d.size()) == 2, ValueError(
            "Left tensor should have a shape of 2D, but had shape {}!".format(
                left_2d.size()
            )
        )
        assert len(right_2d.size()) == 2, ValueError(
            "Right tensor should have a shape of 2D, but had shape {}!".format(
                right_2d.size()
            )
        )
        # Generate the mask.
        left_2d, right_2d = _pad_2Dtensors_if_needed(left_2d, right_2d)
        """
        if left_2d.size(-1) != self._max_dims_or_dims:
            left_2d = self._left_linear_proj(left_2d)
            left_2d = self._left_ln(left_2d) if self._use_layernorm else left_2d
        else:
            self._left_linear_proj = None
            self._left_ln = None
        
        if right_2d.size(-1) != self._max_dims_or_dims:
            right_2d = self._right_linear_proj(right_2d)
            right_2d = self._right_ln(right_2d) if self._use_layernorm else right_2d
        else:
            self._right_linear_proj = None
            self._right_ln = None
        """

        out = left_2d + right_2d
        if out.size(-1) != self._max_dims_or_dims:
            out = self._linear_proj(out)
        else:
            self._linear_proj = None
        if self._use_layernorm:
            out = self._layernorm(out)
        else:
            self._layernorm = None
        # No activation since we are using projection.
        if not self._fixed:
            mask = _mask_generator(self._max_dims_or_dims, dims_in_use, left_2d.device)
            return torch.multiply(out, mask)
        else:
            return out


class LazySelfLinear(nn.Module):
    """
    This module creates a linear transformation, mapping a tensor to the same dimension with a linear layer.
    """
    def __init__(self):
        super().__init__()
        self._linear = None
        self._linear_size: int = -1
    
    def forward(self, x):
        if self._linear is None:
            self._linear = nn.Linear(x.size(-1), x.size(-1), bias=True).to(x.device)
            self._linear_size = x.size(-1)
        assert x.size(-1) == self._linear_size, "'LazySelfLinear' inconsistent size: {} vs {}".format(
            self._linear_size, x.size(-1))
        return self._linear(x)

class SigmoidGating(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Sigmoid gating which processes sigmoid(proj(left)) * proj(right).
        Args:
            :params fixed (bool): Whether fixing this layer or not. If fixed, no masking will be utilized when
            creating sub-networks with this block.
        Kwargs:
            :params use_layernorm (bool): Whether attach layernorm at the end of sigmoid gating.
            :params max_dims_or_dims (int): The maximum dimension to project the output tensor from sigmoid gating,
            or dimension for a fixed subnet.

            Update (09/24/2022): instead of doing right * sigmoid(linear(left)), do linear(right * sigmoid(left)) instead. This helps to enable a better weight sharing and gating.
        """
        super(SigmoidGating, self).__init__()
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._use_layernorm = kwargs["use_layernorm"]
        self._fixed = fixed        
        # Linear on left tensor.
        self._left_self_linear = LazySelfLinear()
        self._linear_proj = nn.LazyLinear(self._max_dims_or_dims, bias=True)
        if self._use_layernorm:
            self._layernorm = nn.LayerNorm(self._max_dims_or_dims)
        else:
            self._layernorm = None

    def forward(self, left_2d: torch.Tensor, right_2d: torch.Tensor, dims_in_use: int):
        assert len(left_2d.size()) == 2, ValueError(
            "Left tensor should have a shape of 2D, but had shape {}!".format(
                left_2d.size()
            )
        )
        assert len(right_2d.size()) == 2, ValueError(
            "Right tensor should have a shape of 2D, but had shape {}!".format(
                right_2d.size()
            )
        )
        if not self._fixed:
            assert dims_in_use <= self._max_dims_or_dims, ValueError(
                "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                    dims_in_use, self._max_dims_or_dims
                )
            )

        # Firstly, do a forward pass on left tensors. Left will always go through a linear.        
        # out_left = torch.sigmoid(self._left_linear(left_2d))
        """
        if right_2d.size(-1) != self._max_dims_or_dims:
            out_right = self._right_linear(right_2d)
        else:
            out_right = right_2d
            # Remove redundant left linear layers if needed.
            self._right_linear = None
        """
        out_left, out_right = _pad_2Dtensors_if_needed(left_2d, right_2d)
        # Self linear in gating.
        out_left = self._left_self_linear(out_left)
        # Sigmoid.
        out_left = torch.sigmoid(out_left)

        out = out_left * out_right
        if out.size(-1) != self._max_dims_or_dims:
            out = self._linear_proj(out)
        else:
            self._linear_proj = None
        # Should not skip layernorm here as gating does not necessarily produce normalized tensors.
        if self._layernorm is not None:
            out = self._layernorm(out)
        if not self._fixed:
            mask = _mask_generator(self._max_dims_or_dims, dims_in_use, left_2d.device)
            # Masking left branch.
            return torch.multiply(out, mask)
        else:
            return out

# NOTE: Important. Please tune manually when transfer to another new application.
LN_INIT = 0.17
class Transformer(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        """
        Transformer Block for 3D inputs. Basically, takes in (Q,K,V) as 3D tensors and output a 3D tensor.
        Args:
            :params fixed (bool): Whether fixing this layer or not. If fixed, no masking will be utilized when
            creating sub-networks with this block.
        Potential Kwargs:
            :params use_layernorm (bool): Whether attach layernorm at the end of sigmoid gating.
            :params activation (str): Activation function.
            :params max_dims_or_dims (int): The maximum dimension to project the output tensor from dot-product,
            or dimension for a fixed subnet.
        """
        super(Transformer, self).__init__()
        self._use_layernorm = kwargs["use_layernorm"]
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._activation = kwargs["activation"]
        self._embedding_dim = kwargs["embedding_dim"]
        # Linear projection first, and then add/ln
        # Linear projection.
        self._linear_proj = nn.LazyLinear(
            self._max_dims_or_dims, bias=not self._use_layernorm,
        )
        self._proj_ln = nn.LayerNorm(self._max_dims_or_dims) if self._use_layernorm else None
        # MHA Layernorm
        self._mha = nn.MultiheadAttention(self._embedding_dim, num_heads=NUM_MHA_HEADS, batch_first=True)
        self._attn_ln = nn.LayerNorm(self._embedding_dim, eps=1e-5)

        # Add two additional FCs.
        self.attn_fc1 = nn.LazyLinear(self._embedding_dim)
        self.attn_fc2 = nn.LazyLinear(self._embedding_dim) 
        self._attn_fc_ln = nn.LayerNorm(self._embedding_dim, eps=1e-5)
        # Dropout to improve regularization.
        self._dropout = kwargs["dropout"] if "dropout" in kwargs else 0.0
        self._activation_fn = _activation_fn_lib[self._activation]
        self._fixed = fixed

        if self._attn_ln is not None:
            torch.nn.init.constant_(self._attn_ln.weight, LN_INIT)

        if self._attn_fc_ln is not None:
            torch.nn.init.constant_(self._attn_fc_ln.weight, LN_INIT)

    def forward(self, sparse_t: torch.Tensor, dims_in_use: int):
        # Sparse_t should be a 3D tensor of [Batch, num_inputs, embed_dims]
        assert len(sparse_t.size()) == 3, ValueError(
            "Input must have a shape of 3D, but had shape {}!".format(sparse_t.size())
        )
        # First, do projection
        sparse_t_proj = self._linear_proj(sparse_t.transpose(1, 2))
        sparse_t_proj = self._proj_ln(sparse_t_proj) if self._proj_ln is not None else sparse_t_proj
        sparse_t_proj = sparse_t_proj.transpose(1, 2)

        # Apply a mask here, masking out the middle dimension.
        if not self._fixed:
            sparse_t_proj = sparse_t_proj.transpose(1, 2)
            mask = (
                _mask_generator(
                    self._max_dims_or_dims, dims_in_use, sparse_t_proj.device)
                if not self._fixed
                else None
            )
            sparse_t_proj = sparse_t_proj * mask            # [B, dim, N]
            sparse_t_proj = sparse_t_proj.transpose(1, 2)

        attn_out, _ = self._mha(sparse_t_proj, sparse_t_proj, sparse_t_proj, need_weights=False)
        # Add + LN
        attn_out = attn_out + sparse_t_proj
        if self._attn_ln is not None:
            attn_out = self._attn_ln(attn_out)

        # Stack two FCs.
        attn_out_fcs = F.relu(self.attn_fc1(attn_out))
        attn_out_fcs = self.attn_fc2(attn_out_fcs)
        attn_out_fcs = attn_out + attn_out_fcs
        if self._attn_fc_ln is not None:
            attn_out_fcs = self._attn_fc_ln(attn_out_fcs)

        # Apply another mask here, masking out the middle dimension. (Remove redundant representation)
        if not self._fixed:
            attn_out_fcs = attn_out_fcs.transpose(1, 2)
            mask = (
                _mask_generator(self._max_dims_or_dims, dims_in_use, attn_out_fcs.device)
                if not self._fixed
                else None
            )
            attn_out_fcs = attn_out_fcs * mask            # [B, dim, N]
            attn_out_fcs = attn_out_fcs.transpose(1, 2)

        return attn_out_fcs


class Zeros3D(nn.Module):
    def __init__(self, **kwargs):
        """
        Self attention for 3D inputs. Basically, takes in (Q,K,V) as 3D tensors and output a 3D tensor.
        Args:
        Potential Kwargs:
            :params max_dims_or_dims (int): The maximum dimension to project the output tensor from elastic linear,
            or dimension for a fixed subnet. (Useless)
        """
        super(Zeros3D, self).__init__()
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]

    def forward(self, sparse_t: torch.Tensor, dims_in_use: int):
        # Note: 'in_dims' here is useless. It is only used for sanity check.
        assert len(sparse_t.size()) == 3, ValueError(
            "Input must have a shape of 3D, but had shape {}!".format(sparse_t.size())
        )

        assert dims_in_use <= self._max_dims_or_dims, ValueError(
            "If not in fixed mode where supernet is trained, \
           'dims_in_use' should always be smaller than 'max_dims_or_dims', but found {} vs {}! ".format(
                dims_in_use, self._max_dims_or_dims
            )
        )
        return _zeros_generator(
            torch.Size((sparse_t.size(0), self._max_dims_or_dims, sparse_t.size(2))),
            sparse_t.device,
        )

class FactorizationMachine3D(nn.Module):
    def __init__(self, fixed: bool = False, **kwargs):
        super(FactorizationMachine3D, self).__init__()
        self._use_layernorm = kwargs["use_layernorm"]
        self._max_dims_or_dims = kwargs["max_dims_or_dims"]
        self._linear_proj = nn.LazyLinear(
            self._max_dims_or_dims, bias=not self._use_layernorm
        )
        self._fixed = fixed
        if self._use_layernorm:
            self._linear_layernorm = nn.LayerNorm(self._max_dims_or_dims, eps=1e-5)
        pass

    def forward(self, sparse_t: torch.Tensor, dims_in_use: int):
        assert len(sparse_t.size()) == 3, "Tensor must be a sparse tensor!"
        # [batch, num_inputs, embeddings]
        square_of_sum = torch.sum(sparse_t, dim=1) ** 2
        sum_of_square = torch.sum(sparse_t ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if ix.size(-1) != self._max_dims_or_dims:
            ix = self._linear_proj(ix)
            ix = self._linear_layernorm(ix) if self._use_layernorm else ix
        else:
            self._linear_proj, self._use_layernorm = None, None
        # Mask both tensors as they have limited dimensions.
        mask = (
            _mask_generator(self._max_dims_or_dims, dims_in_use, ix.device) if not self._fixed
            else None
        )
        out = torch.multiply(ix, mask) if not self._fixed else ix
        return out

# Note: Factorization machine in the form of 2D is reduced to identity mapping.
