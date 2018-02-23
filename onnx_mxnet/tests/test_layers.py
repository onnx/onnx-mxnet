# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Tests for individual operators"""
from __future__ import absolute_import
import unittest
import numpy as np
import numpy.testing as npt
from onnx import helper
from onnx_mxnet import backend as mxnet_backend

class TestLayers(unittest.TestCase):
    """Tests for different layers comparing output with numpy operators.
    [WIP] More tests coming soon!"""

    def _random_array(self, shape):
        """Generate random array according to input shape"""
        return np.random.ranf(shape).astype("float32")

    def test_abs(self):
        """Test for abs operator"""
        node_def = helper.make_node("Abs", ["ip1"], ["ip2"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.abs(ip1))

    def test_add(self):
        """Test for add operator with/without broadcasting"""
        node_def = helper.make_node("Add", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.add(ip1, ip2))

        node_def = helper.make_node("Add", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 16, 16])
        ip2 = self._random_array([1, 16, 16])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.add(ip1, ip2))

    def test_sum(self):
        """Test for sum operator"""
        node_def = helper.make_node("Sum", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.add(ip1, ip2))

    def test_sub(self):
        """Test for sub operator with/without broadcasting"""
        node_def = helper.make_node("Sub", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.subtract(ip1, ip2))

        node_def = helper.make_node("Sub", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.subtract(ip1, ip2))

    def test_mul(self):
        """Test for mul operator with/without broadcasting"""
        node_def = helper.make_node("Mul", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.multiply(ip1, ip2))

        node_def = helper.make_node("Mul", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.multiply(ip1, ip2))

    def test_div(self):
        """Test for div operator with/without broadcasting"""
        node_def = helper.make_node("Div", ["ip1", "ip2"], ["op1"], broadcast=1)
        ip1 = self._random_array([1, 1, 5, 5])
        ip2 = self._random_array([5])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.divide(ip1, ip2))

        node_def = helper.make_node("Div", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 1, 16, 16])
        ip2 = self._random_array([1, 1, 16, 16])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.divide(ip1, ip2))

    def test_relu(self):
        """Test for relu operator"""
        node_def = helper.make_node("Relu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 256])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.maximum(ip1, 0))

    def test_neg(self):
        """Test for neg operator"""
        node_def = helper.make_node("Neg", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.negative(ip1))

    def test_reciprocal(self):
        """Test for reciprocal operator"""
        node_def = helper.make_node("Reciprocal", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.reciprocal(ip1))

    def test_floor(self):
        """Test for floor operator"""
        node_def = helper.make_node("Floor", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.floor(ip1))

    def test_ceil(self):
        """Test for ceil operator"""
        node_def = helper.make_node("Ceil", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.ceil(ip1))

    def test_sqrt(self):
        """Test for sqrt operator"""
        node_def = helper.make_node("Sqrt", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 1000])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.sqrt(ip1))

    def test_leaky_relu(self):
        """Test for LeakyRelu operator"""
        node_def = helper.make_node("LeakyRelu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0][0]
        # default slope in leakyrelu is 0.25
        numpy_output = [x if x > 0 else x*0.25 for x in ip1[0]]
        npt.assert_almost_equal(output, numpy_output)

    def test_elu(self):
        """Test for elu operator"""
        node_def = helper.make_node("Elu", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0][0]
        # default slope in elu is 0.25
        numpy_output = [x if x > 0 else (np.exp(x)-1)*0.25 for x in ip1[0]]
        npt.assert_almost_equal(output, numpy_output)

    def test_exp(self):
        """Test for exp operator"""
        node_def = helper.make_node("Exp", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.exp(ip1))

    def test_log(self):
        """Test for log operator"""
        node_def = helper.make_node("Log", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.log(ip1))

    def test_tanh(self):
        """Test for tanh operator"""
        node_def = helper.make_node("Tanh", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.tanh(ip1))

    def test_pow(self):
        """Test for pow operator"""
        node_def = helper.make_node("Pow", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.power(ip1, ip2))

    def test_sigmoid(self):
        """Test for sigmoid operator"""
        node_def = helper.make_node("Sigmoid", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])
        np_output = [1/(1+np.exp(-ip1))]
        npt.assert_almost_equal(output, np_output)

    def test_maximum(self):
        """Test for maximum operator"""
        node_def = helper.make_node("Max", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.maximum(ip1, ip2))

    def test_minimum(self):
        """Test for minimum operator"""
        node_def = helper.make_node("Min", ["ip1", "ip2"], ["op1"])
        ip1 = self._random_array([1, 10])
        ip2 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1, ip2])[0]
        npt.assert_almost_equal(output, np.minimum(ip1, ip2))

    def test_softmax(self):
        """Test for softmax operator"""
        node_def = helper.make_node("Softmax", ["ip1"], ["op1"])
        ip1 = self._random_array([1, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        exp_score = np.exp(ip1)
        numpy_op = exp_score / exp_score.sum(0)
        npt.assert_almost_equal(output, numpy_op)

    def test_reduce_max(self):
        """Test for ReduceMax operator"""
        node_def = helper.make_node("ReduceMax", ["ip1"], ["op1"], axes=[1, 0], keepdims=1)
        ip1 = self._random_array([3, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        numpy_op = np.max(ip1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op)

    def test_reduce_min(self):
        """Test for ReduceMin operator"""
        node_def = helper.make_node("ReduceMin", ["ip1"], ["op1"], axes=[1, 0], keepdims=1)
        ip1 = self._random_array([3, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        numpy_op = np.min(ip1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op)

    def test_reduce_sum(self):
        """Test for ReduceSum operator"""
        node_def = helper.make_node("ReduceSum", ["ip1"], ["op1"], axes=[1, 0], keepdims=1)
        ip1 = self._random_array([3, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        numpy_op = np.sum(ip1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    def test_reduce_mean(self):
        """Test for ReduceMean operator"""
        node_def = helper.make_node("ReduceMean", ["ip1"], ["op1"], axes=[1, 0], keepdims=1)
        ip1 = self._random_array([3, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        numpy_op = np.mean(ip1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    def test_reduce_prod(self):
        """Test for ReduceProd operator"""
        node_def = helper.make_node("ReduceProd", ["ip1"], ["op1"], axes=[1, 0], keepdims=1)
        ip1 = self._random_array([3, 10])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        numpy_op = np.prod(ip1, axis=(1, 0), keepdims=True)
        npt.assert_almost_equal(output, numpy_op, decimal=5)

    def test_squeeze(self):
        """Test for squeeze operator"""
        node_def = helper.make_node("Squeeze", ["ip1"], ["op1"], axes=[1, 3])
        ip1 = self._random_array([3, 1, 2, 1, 4])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.squeeze(ip1, axis=[1, 3]))

    def test_reshape(self):
        """Test for reshape operator"""
        node_def = helper.make_node("Reshape", ["ip1"], ["op1"], shape=[3, 2, 2, 2])
        ip1 = self._random_array([2, 3, 4])
        output = mxnet_backend.run_node(node_def, [ip1])[0]
        npt.assert_almost_equal(output, np.reshape(ip1, [3, 2, 2, 2]))

if __name__ == '__main__':
    unittest.main()
