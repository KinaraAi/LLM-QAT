# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, group_size=-1 ,type="weight",quantization_params = {}):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        shape = input.shape
        if type == "weight" and 'weight_scales' in quantization_params and quantization_params['weight_scales'] is not None:
            s = quantization_params['weight_scales']
        else: 
            if layerwise:
                max_input = torch.max(torch.abs(input)).expand_as(input)
            else:
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    if group_size > 0:
                        input = input.reshape(-1, group_size)
                    max_input = (
                        torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                        .expand_as(input)
                        .detach()
                    )
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    max_input = (
                        torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                        .unsqueeze(-1)
                        .expand_as(input)
                        .detach()
                    )
                else:
                    raise ValueError
            s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
            if type == "weight" and 'weight_scales' in quantization_params and quantization_params['wt_scale_freeze'] and quantization_params['weight_scales'] is None:
                quantization_params['weight_scales'] = s
        output = torch.round(input * s).div(s + 1e-6)
        if group_size > 0:
            output = output.reshape(shape)
        return output if "weight_scales" not in quantization_params else (output, s)

    @staticmethod
    def backward(ctx, grad_output, group_size=-1 ,type="weight" ,params=None):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None, None,None


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=32,
        a_bits=32,
        group_size=-1,
        act_layerwise=False,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        self.quantization_params = {'weight_scales': None, 'activation_scales':None,'wt_scale_freeze': False, 'act_scale_freeze': False} 
        self.group_size = group_size
        # params for weight quant
        # if self.w_bits < 32:
        #     self.weight_clip_val = Parameter(torch.tensor([-2.0, 2.0]), requires_grad=False)
        if self.a_bits < 32 and self.a_bits > 2:
            if symmetric:
                self.act_quantizer = SymQuantizer
            else:
                self.act_quantizer = AsymQuantizer

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 3:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            weight, scales = SymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise, self.group_size ,"weight", self.quantization_params
            )
            if self.quantization_params['wt_scale_freeze'] and self.quantization_params['weight_scales'] is None:
                self.quantization_params['weight_scales'] = scales
        else:
            if self.w_bits == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = torch.mean(
                        abs(real_weights), dim=1, keepdim=True
                    ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
                weight = (
                    quan_weights_no_grad.detach() - real_weights.detach() + real_weights
                )
            # elif self.w_bits == 2:
            #     scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            #     quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
            else:
                dim1, dim2 = real_weights.shape
                if self.group_size > 0:
                    dim1, dim2 = real_weights.shape
                    real_weights = real_weights.reshape(-1, self.group_size)
                num_bits = 2 ** (self.w_bits - 1)
                clip_val = 1 - 1e-2
                if self.quantization_params['wt_scale_freeze'] and self.quantization_params['weight_scales'] is not None:
                    scaling_factor = self.quantization_params['weight_scales']
                else:
                    if self.weight_layerwise:
                        scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
                    else:
                        scaling_factor = (
                            2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                        )
                    if self.quantization_params['wt_scale_freeze'] and self.quantization_params['weight_scales'] is None:
                        self.quantization_params['weight_scales'] = scaling_factor
                quan_weights_no_grad = (
                    scaling_factor
                    * (
                        torch.round(
                            torch.clamp(
                                real_weights / scaling_factor, -clip_val, clip_val
                            )
                            * num_bits
                            - 0.5
                        )
                        + 0.5
                    )
                    / num_bits
                )

                weight = (
                    quan_weights_no_grad.detach() - real_weights.detach() + real_weights
                )
                if self.group_size > 0:
                    weight = weight.reshape(dim1, dim2)
        # Quantize inputs
        if self.a_bits < 32 and self.a_bits > 2:
            act_clip_val = torch.tensor([-2.0, 2.0])
            input_ = self.act_quantizer.apply(
                input_, act_clip_val, self.a_bits, self.act_layerwise
            )

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
