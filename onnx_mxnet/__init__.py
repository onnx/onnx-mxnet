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
"""import function"""
import onnx
from .import_onnx import GraphProto

def import_model(model_file):
    """Imports the supplied ONNX model file into MXNet symbol and parameters.

    Parameters
    ----------
    model_file : ONNX model file name

    Returns
    -------
    sym : mx.symbol
        Compatible mxnet symbol

    params : dict of str to mx.ndarray
        Dict of converted parameters stored in mx.ndarray format
    """
    graph = GraphProto()

    # loads model file and returns ONNX protobuf object
    model_proto = onnx.load(model_file)
    sym, params = graph.from_onnx(model_proto.graph)
    return sym, params
