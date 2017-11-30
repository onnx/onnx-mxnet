# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Derived from Apache 2.0 licensed onnx.py file from DMLC NNVM:
# https://github.com/dmlc/nnvm/blob/3da53e46db57c438b05fbebe8aa332ee8c5994d1/python/nnvm/frontend/onnx.py

# coding: utf-8

from onnx_mxnet.common import Renamer, AttributeConverter as AttrCvt

def _revert_caffe2_pad(attr):
    """Removing extra padding from Caffe2."""
    if len(attr) == 4:
        attr = attr[:2]
    elif len(attr) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(attr))
    return attr

def _math_name_picker(surfix):
    def _impl(attr):
        if attr.get('broadcast', 0):
            return 'broadcast_' + surfix
        return 'elemwise_' + surfix
    return _impl

def _broadcast_constraint():
    def _broadcast_check(attrs):
        if attrs.get('axis', None):
            return False
        return True
    return _broadcast_check, "Specifying broadcast axis not allowed."

# checking dimensions for conv, deconv, pooling operators
def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False
    return _dim_check, "Only 2d kernel supported."

# converting attributes for add operator
def _elemwise(name):
    return AttrCvt(
        op_name=_math_name_picker(name),
        disables=['axis'],
        ignores=['broadcast'])

# converting attributes for pooling operator
def _pooling(name):
    return AttrCvt(
        op_name='Pooling',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'pads': ('pad', (0, 0), _revert_caffe2_pad)},
        # pooling convention full to match caffe2
        extras={'pool_type': name, 'pooling_convention':'full'},
        ignores=['dilations'],
        custom_check=_dimension_constraint())

# converting attributes for convolution operator
def _conv():
    return AttrCvt(
        op_name='Convolution',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0,0)),
            'pads': ('pad', (0,0), _revert_caffe2_pad),
            'group': ('num_group', 1)},
        custom_check=_dimension_constraint())

# converting attributes for deconvolution operator
def _conv_transpose():
    return AttrCvt(
        op_name='Deconvolution',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0, 0)),
            'pads': ('pad', (0, 0), _revert_caffe2_pad)},
        disables=['output_shape'],
        custom_check=_dimension_constraint())

def _change_eps_cudnn(attr):
    """Limiting eps value to 1e-5 for cudnn batchnorm."""
    if attr < 1e-5:
        attr = 1e-4
    return attr

# converting attributes for BatchNorm operator
def _batch_norm():
    return AttrCvt(
        op_name='BatchNorm',
        transforms={'epsilon': ('eps', (1e-5), _change_eps_cudnn)},
        ignores=['spatial', 'is_test','consumed_inputs'])

# converting attributes for LeakyRelu operator
def _activation(name):
    return AttrCvt(
        op_name='LeakyReLU',
        transforms={
            'alpha':'slope'},
        extras={'act_type': name})

def _pad_sequence_fix(attr):
    new_attr = ()
    if len(attr)%2==0:
        for index in range(len(attr) / 2):
            new_attr = new_attr + attr[index::len(attr) / 2]
    return new_attr

# converting attributes for Pad operator
def _pad():
    return AttrCvt(
        op_name='pad',
        transforms={
            'pads': ('pad_width', (0,0,0,0,0,0,0,0),_pad_sequence_fix),
            'value':'constant_value'})

# Requires kernel attribute which is not present in onnx currently. So for now giving default kernel.
def _global_pooling(name):
    return AttrCvt(
        op_name='Pooling',
        extras={'global_pool': True,
                'kernel': (1,1),
                'pool_type': name})

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
_convert_map = {
    # defs/experimental
    'FC'            : AttrCvt('FullyConnected', ignores=['axis', 'axis_w']),

    # defs/generator
    'RandomUniform' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormal'  : AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),
    'RandomUniformLike' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormalLike': AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),

    # defs/logical

    # defs/math
    'Add'           : _elemwise('add'),
    'Sub'           : _elemwise('sub'),
    'Mul'           : _elemwise('mul'),
    'Div'           : _elemwise('div'),
    'Neg'           : Renamer('negative'),
    'Abs'           : Renamer('abs'),
    'Reciprocal'    : Renamer('reciprocal'),
    'Floor'         : Renamer('floor'),
    'Ceil'          : Renamer('ceil'),
    'Sqrt'          : Renamer('sqrt'),
    'Gemm'          : AttrCvt('linalg_gemm', {'transA':'transpose_a', 'transB':'transpose_b'}, ignores=['broadcast']),
    'Relu'          : Renamer('relu'),
    'LeakyRelu'     : AttrCvt('LeakyReLU', {'alpha', 'slope'}),
    # 'Selu'
    'Elu'           : _activation('elu'),
    'Exp'           : Renamer('exp'),
    'Log'           : Renamer('log'),
    'Tanh'          : Renamer('tanh'),
    'Pow'           : AttrCvt('pow', {'exponent':'exp'}),
    'Dot'           : Renamer('dot'),
    'MatMul'        : Renamer('linalg_gemm2'),
    # 'PRelu'
    'Sigmoid'       : Renamer('sigmoid'),
    'Max'           : Renamer('maximum'), #elemwise maximum
    'Min'           : Renamer('minimum'), #elemwise minimum
    'Sum'           : Renamer('add_n'), #elemwise sum
    # softmax default axis is different in onnx
    'Softmax'       : AttrCvt('softmax', extras={'axis': 1}),

    # defs/nn
    'AveragePool'   : _pooling('avg'),
    'MaxPool'       : _pooling('max'),
    'Conv'          : _conv(),
    'ConvTranspose' : _conv_transpose(),
    'GlobalAveragePool': _global_pooling('avg'),
    'GlobalMaxPool' : _global_pooling('max'),
    'BatchNormalization': _batch_norm(),
    'SpatialBN'     : _batch_norm(),
    'Dropout'       : AttrCvt('Dropout', {'ratio': 'p'}, ignores=['is_test']),
    'Flatten'       : Renamer('Flatten'),
    'LRN'           : AttrCvt('LRN', {'bias': 'knorm', 'size' : 'nsize'}),
    # defs/reduction
    'ReduceMax'     : AttrCvt('max', {'axes', 'axis'}),
    'ReduceMin'     : AttrCvt('min', {'axes', 'axis'}),
    'ReduceSum'     : AttrCvt('sum', {'axes', 'axis'}),
    'ReduceMean'    : AttrCvt('mean', {'axes', 'axis'}),
    'ReduceProd'    : AttrCvt('prod', {'axes', 'axis'}),
    # 'ReduceLogSumExp'
    # 'ArgMax'        : _arg_op('argmax'),
    # 'ArgMin'        : _arg_op('argmin'),

    # defs/tensor
    'Cast'          : AttrCvt('cast', {'to': 'dtype'}),
    'Reshape'       : AttrCvt('reshape', {'shape': 'shape'}),
    'Concat'        : AttrCvt('concat', {'axis': 'dim'}),
    'Split'         : AttrCvt('split', {'split': 'num_outputs'}),
    'Pad'           : _pad(),
    'Slice'         : AttrCvt('slice_axis', {'axes': 'axis', 'ends': 'end', 'starts': 'begin'}),
    'Transpose'     : AttrCvt('transpose', {'perm': 'axes'}),
    # 'Gather'
    # 'Squeeze'
}
