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
from __future__ import absolute_import as _abs
import mxnet as mx
import numpy as np
import os
import numpy.testing as npt

import onnx_mxnet
from collections import namedtuple
import tarfile

URLS = {
'squeezenet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz',
'shufflenet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/shufflenet.tar.gz',
'inception_v1_onnx' : 'https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz',
'inception_v2_onnx' : 'https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz',
'bvlc_alexnet_onnx' : 'https://s3.amazonaws.com/download.onnx/models/bvlc_alexnet.tar.gz',
'densenet121_onnx' : 'https://s3.amazonaws.com/download.onnx/models/densenet121.tar.gz',
'resnet50_onnx' : 'https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz',
'vgg16_onnx' : 'https://s3.amazonaws.com/download.onnx/models/vgg16.tar.gz',
'vgg19_onnx' : 'https://s3.amazonaws.com/download.onnx/models/vgg19.tar.gz'
}

# load protobuf format
def _as_abs_path(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir, fname)

# download test image
def download(url, path, overwrite=False):
    import urllib2, os
    if os.path.exists(path) and not overwrite:
        return
    print('Downloading {} to {}.'.format(url, path))
    with open(path, 'w') as f:
        f.write(urllib2.urlopen(url).read())

def extract_file(model_tar):
    # extract tar file
    tar = tarfile.open(model_tar, "r:gz")
    tar.extractall()
    tar.close()
    path = model_tar.rsplit('_', 1)[0]
    # return model, inputs, outputs path
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(cur_dir, path, 'model.pb')
    npz_path = os.path.join(cur_dir, path, 'test_data_0.npz')
    sample = np.load(npz_path, encoding='bytes')
    input_data = list(sample['inputs'])
    output_data = list(sample['outputs'])
    return model_path, input_data, output_data

def verify_onnx_forward_impl(model_path, input_data, output_data):
    print "Converting onnx format to mxnet's symbol and params..."
    sym, params = onnx_mxnet.import_model(model_path)
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('input_0', input_data.shape)], label_shapes=None)
    mod.set_params(arg_params=params, aux_params=None)
    # run inference
    Batch = namedtuple('Batch', ['data'])

    mod.forward(Batch([mx.nd.array(input_data)]))

    # Run the model with an onnx backend and verify the results
    npt.assert_equal(mod.get_outputs()[0].shape, output_data.shape)
    npt.assert_almost_equal(output_data, mod.get_outputs()[0].asnumpy(), decimal=3)
    print "Conversion Successful"

def verify_model(name):
    print "Testing model ", name
    download(URLS.get(name), name, False)
    model_path, inputs, outputs = extract_file(name)
    input_data = np.asarray(inputs[0], dtype=np.float32)
    output_data = np.asarray(outputs[0], dtype=np.float32)
    verify_onnx_forward_impl(model_path, input_data, output_data)

if __name__ == '__main__':
    verify_model('squeezenet_onnx') # working
    verify_model('bvlc_alexnet_onnx') # working
    verify_model('vgg16_onnx') # working
    verify_model('vgg19_onnx')  # working
    # verify_model('inception_v1_onnx') # working, accuracy is different
    # verify_model('inception_v2_onnx') # [WIP]
    # verify_model('shufflenet_onnx') # [WIP]
    # verify_model('densenet121_onnx') # [WIP]
    # verify_model('resnet50_onnx') # [WIP]
