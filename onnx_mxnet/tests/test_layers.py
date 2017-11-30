# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import absolute_import

import unittest
import numpy as np
import numpy.testing as npt
from onnx import helper
import onnx_mxnet

class TestLayers(unittest.TestCase):
    """Tests for different layers comparing output with numpy operators.
    [WIP] More tests coming soon!"""

    def _random_array(self,shape):
        return np.random.ranf(shape).astype("float32")

    def test_abs(self):
        node_def = helper.make_node("Abs", ["ip1"], ["ip2"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.abs(ip1))

    def test_add(self):
        node_def = helper.make_node("Add", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.add(ip1, ip2))

        node_def = helper.make_node("Add", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 16, 16])
        ip2 = self._random_array([1, 16, 16])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.add(ip1, ip2))

    def test_sum(self):
        node_def = helper.make_node("Sum", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.add(ip1, ip2))

    def test_sub(self):
        node_def = helper.make_node("Sub", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.subtract(ip1, ip2))

        node_def = helper.make_node("Sub", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.subtract(ip1, ip2))

    def test_mul(self):
        node_def = helper.make_node("Mul", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.multiply(ip1, ip2))

        node_def = helper.make_node("Mul", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.multiply(ip1, ip2))

    def test_div(self):
        node_def = helper.make_node("Div", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.divide(ip1, ip2))

        node_def = helper.make_node("Div", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.divide(ip1, ip2))

    def test_concat(self):
        node_def = helper.make_node("Concat", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 3, 128, 256])
        ip2 = self._random_array([1, 3, 128, 256])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0].asnumpy()
        npt.assert_almost_equal(output, np.concatenate((ip1, ip2), axis=1))

    def test_relu(self):
        node_def = helper.make_node("Relu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 256])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.maximum(ip1, 0))

    def test_neg(self):
        node_def = helper.make_node("Neg", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.negative(ip1))

    def test_reciprocal(self):
        node_def = helper.make_node("Reciprocal", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.reciprocal(ip1))

    def test_floor(self):
        node_def = helper.make_node("Floor", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.floor(ip1))

    def test_ceil(self):
        node_def = helper.make_node("Ceil", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.ceil(ip1))

    def test_sqrt(self):
        node_def = helper.make_node("Sqrt", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.sqrt(ip1))

    def test_leaky_relu(self):
        node_def = helper.make_node("LeakyRelu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0][0].asnumpy()
        # default slope in leakyrelu is 0.25
        numpy_output = [x if x>0 else x*0.25 for x in ip1[0]]
        npt.assert_almost_equal(output, numpy_output)

    def test_elu(self):
        node_def = helper.make_node("Elu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0][0].asnumpy()
        # default slope in elu is 0.25
        numpy_output = [x if x>0 else (np.exp(x)-1)*0.25 for x in ip1[0]]
        npt.assert_almost_equal(output, numpy_output)

    def test_exp(self):
        node_def = helper.make_node("Exp", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.exp(ip1))

    def test_log(self):
        node_def = helper.make_node("Log", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.log(ip1))

    def test_tanh(self):
        node_def = helper.make_node("Tanh", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.tanh(ip1))

    def test_pow(self):
        node_def = helper.make_node("Pow", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.power(ip1, ip2))

    def test_sigmoid(self):
        node_def = helper.make_node("Sigmoid", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0].asnumpy()
        np_output = [1/(1+np.exp(-ip1))]
        npt.assert_almost_equal(output, np_output)

    def test_maximum(self):
        node_def = helper.make_node("Max", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.maximum(ip1, ip2))

    def test_minimum(self):
        node_def = helper.make_node("Min", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1, ip2])[0][0].asnumpy()
        npt.assert_almost_equal(output, np.minimum(ip1, ip2))

    def test_softmax(self):
        node_def = helper.make_node("Softmax", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = onnx_mxnet.run_node(node_def, [ip1])[0][0].asnumpy()
        exp_score = np.exp(ip1)
        numpy_op = exp_score / exp_score.sum(0)
        npt.assert_almost_equal(output, numpy_op)

if __name__ == '__main__':
    unittest.main()
