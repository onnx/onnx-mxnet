# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# pylint: disable=invalid-name, exec-used
"""Setup onnx-mxnet package"""
from setuptools import setup, find_packages

pkgs = find_packages()

setup(
    name='onnx-mxnet',
    version='0.4.1',
    description='ONNX-MXNet Model converter',
    url='https://github.com/onnx/onnx-mxnet',
    keywords='ONNX MXNet model converter deep learning',
    packages=pkgs,
    install_requires=['mxnet>=0.11.0', 'onnx>=1.0.1'],
    tests_require=['pytest', 'pylint'],
    include_package_data=True,
    license='Apache 2.0'
)
