# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# coding: utf-8
from .import_onnx import GraphProto
import mxnet as mx
import numpy as np
from onnx.backend.base import Backend, BackendRep
from collections import namedtuple

"""Using these functions for onnx test infrastructure."""
class MXNetBackend(Backend):
    @classmethod
    def run_node(cls, node, inputs, device='CPU'):
        """Running individual node inference on mxnet engine and
        return the result to onnx test infrastructure."""
        g = GraphProto()
        sym = g._run_node(node)
        data_names = [i for i in node.input]
        data_shapes = []
        # Adding extra dimension of batch_size 1 if the batch_size is different for multiple inputs.
        for idx, input_name in enumerate(data_names):
            batch_size = 1L
            if len(inputs[idx].shape) < 4 and len(inputs) > 1 and len(set(x.shape[0] for x in inputs)) != 1:
                tuples = ((batch_size,), inputs[idx].shape)
                new_shape = sum(tuples, ())
                data_shapes.append((input_name, new_shape))
            else:
                data_shapes.append((input_name, inputs[idx].shape))

        # create a module
        mod = mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)
        # initializing parameters for calculating result of each individual node
        mod.init_params()
        from collections import namedtuple
        Batch = namedtuple('Batch', ['data'])
        data_forward = []
        for v in inputs:
            # slice and pad operator tests needs 1 less dimension in forward pass
            # otherwise it will throw an error.
            if node.op_type=='Slice' or node.op_type=='Pad':
                data_forward.append(mx.nd.array(v))
            else:
                data_forward.append(mx.nd.array([v]))

        mod.forward(Batch(data_forward))
        result = mod.get_outputs()[0].asnumpy()
        if node.op_type=='Slice' or node.op_type=='Pad':
            return [result]
        return result

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        """For running end to end model(used for onnx test backend)"""
        graph = GraphProto()
        sym, params = graph.from_onnx(model.graph)
        return MXNetBackendRep(sym, params)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'


class MXNetBackendRep(BackendRep):
    """Running model inference on mxnet engine and return the result
     to onnx test infrastructure for comparison."""
    def __init__(self, mxnet_symbol, params):
        self.model = mxnet_symbol
        self.params = params

    def run(self, inputs, **kwargs):
        """Run model inference and return the result"""
        input_data = np.asarray(inputs[0], dtype=np.float32)
        # create module
        mod = mx.mod.Module(symbol=self.model, data_names=['input_0'], context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('input_0', input_data.shape)], label_shapes=None)
        mod.set_params(arg_params=self.params, aux_params=None)
        # run inference
        Batch = namedtuple('Batch', ['data'])

        mod.forward(Batch([mx.nd.array(input_data)]))
        result = mod.get_outputs()[0].asnumpy()
        return [result]

prepare = MXNetBackend.prepare

run_node = MXNetBackend.run_node

supports_device = MXNetBackend.supports_device
