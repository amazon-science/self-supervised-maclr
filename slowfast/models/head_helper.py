#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        from detectron2.layers import ROIAlign
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d(
                [pool_size[pathway][0], 1, 1], stride=1
            )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        normalize=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.normalize = normalize

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        # check if audio pathway is compatible with visual ones
        if len(pool_out) > 2:
            a_H, a_W = pool_out[2].size(-2), pool_out[2].size(-1)
            v_H, v_W = pool_out[0].size(-2), pool_out[0].size(-1)
            if a_H != v_H or a_W != v_W:
                assert v_H % a_H == 0 and v_W % a_W == 0, \
                    'Visual pool output should be divisible by audio pool output size'
                a_N, a_C, a_T, _, _ = pool_out[2].shape
                pool_out[2] = pool_out[2].expand([a_N, a_C, a_T, v_H, v_W])
        
        # pool_out[1] = torch.mean(pool_out[1], dim=[2, 3, 4], keepdim=True).repeat(1, 1, pool_out[0].size(2), pool_out[0].size(3), pool_out[0].size(4))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # Optionally L2 normalize x.
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        # Project to label space.
        x = self.projection(x)
        # Performs fully convlutional inference.
        if not self.training:
            if self.act is not None:
                x = self.act(x)
        x = x.mean([1, 2, 3])
        # if not self.training:
        #     if self.act is not None:
        #         x = self.act(x)
        #     x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class FullyConvHead(nn.Module):
    
    def __init__(
        self,
        dim_in,
        num_classes,
        visual_pool_size,
        audio_win,
        act_func="softmax",
    ):
        super(FullyConvHead, self).__init__()
        self.audio_win = audio_win

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Create a pooler.
        if visual_pool_size is None:
            avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            avg_pool = nn.AvgPool3d(visual_pool_size, stride=1)
        self.add_module("pathway0_avgpool", avg_pool)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=2)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        T = inputs[0].size(2)
        pool_out = []
        
        m = getattr(self, "pathway0_avgpool")
        pool_out.append(m(inputs[0]))

        if len(inputs) > 1:
            # (N, C, 1, T, F) -> (N, C, T, 1, F).
            inputs[1] = inputs[1].permute((0, 1, 3, 2, 4))
            inputs[1] = torch.mean(inputs[1], dim=[3, 4])
            inputs[1] = nn.functional.avg_pool1d(inputs[1], kernel_size=self.audio_win, stride=1, padding=self.audio_win//2)
            inputs[1] = nn.functional.interpolate(inputs[1], size=T)
            inputs[1] = inputs[1][:, :, :, None, None] # (N, C, T) -> (N, C, T, 1, 1).
            pool_out.append(inputs[1])

        # check if audio pathway is compatible with visual ones
        if len(pool_out) > 1:
            a_H, a_W = pool_out[1].size(-2), pool_out[1].size(-1)
            v_H, v_W = pool_out[0].size(-2), pool_out[0].size(-1)
            if a_H != v_H or a_W != v_W:
                assert v_H % a_H == 0 and v_W % a_W == 0, \
                    'Visual pool output should be divisible by audio pool output size'
                a_N, a_C, a_T, _, _ = pool_out[1].shape
                pool_out[1] = pool_out[1].expand([a_N, a_C, a_T, v_H, v_W])

        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        # Project to class label space.
        x = self.projection(x)

        # (N, T, H, W, C) -> (N, T, C).
        x = torch.mean(x, dim=[2, 3])
        # x = torch.max(x, dim=2)[0]
        # x = torch.max(x, dim=2)[0]

        # Performs fully convlutional inference.
        if not self.training and self.act is not None:
            x = self.act(x)

        return x


class ContrastiveCodeHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
    ):
        super(ContrastiveCodeHead, self).__init__()
        self.num_pathways = len(dim_in)
        self.num_layers = num_layers
        for pathway in range(self.num_pathways):
            # Perform FC in a fully convolutional manner. The FC layer will be
            # initialized with a different std comparing to convolutional layers.
            cur_dim_in = dim_in[pathway]
            for idx in range(self.num_layers):
                cur_dim_out = dim_out if idx == self.num_layers - 1 else dim_hidden
                projection = nn.Linear(cur_dim_in, cur_dim_out, bias=True)
                self.add_module("pathway{}_projection_{}".format(pathway, idx), projection) 
                cur_dim_in = cur_dim_out
        

    def forward(self, inputs):
        tensor_io = False
        if not isinstance(inputs, (List, Tuple)):
            inputs = [inputs]
            tensor_io = True

        outputs = []
        for pathway, x in enumerate(inputs):
            if x.ndim >= 5:
                x = torch.mean(x, (2, 3, 4), keepdim=False)
            for idx in range(self.num_layers):
                m = getattr(self, "pathway{}_projection_{}".format(pathway, idx))
                x = m(x)
                if idx != self.num_layers - 1:
                    x = F.relu(x)
            x = F.normalize(x, dim=-1)
            outputs.append(x)
        
        if tensor_io:
            outputs = outputs[0]

        return outputs