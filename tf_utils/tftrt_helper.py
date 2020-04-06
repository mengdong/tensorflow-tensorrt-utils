# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import os
import argparse
import numpy as np
import tensorflow as tf
from tasks.image_classification.preprocessing import inception_preprocessing, vgg_preprocessing
from tensorflow.python.client import device_lib
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def get_trt_conversion_params(max_workspace_size_bytes,
                              precision_mode,
                              minimum_segment_size,
                              max_batch_size,
                              use_trt_dynamic_op):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=max_workspace_size_bytes)
    conversion_params = conversion_params._replace(
        precision_mode=precision_mode)
    conversion_params = conversion_params._replace(
        minimum_segment_size=minimum_segment_size)
    conversion_params = conversion_params._replace(
        use_calibration=precision_mode == 'INT8')
    conversion_params = conversion_params._replace(
        max_batch_size=max_batch_size)
    conversion_params = conversion_params._replace(
        is_dynamic_op=use_trt_dynamic_op)
    return conversion_params

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                   'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                   'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                   'image/object/bbox/xmin': tf.io.VarLenFeature(
                       dtype=tf.float32),
                   'image/object/bbox/ymin': tf.io.VarLenFeature(
                       dtype=tf.float32),
                   'image/object/bbox/xmax': tf.io.VarLenFeature(
                       dtype=tf.float32),
                   'image/object/bbox/ymax': tf.io.VarLenFeature(
                       dtype=tf.float32)}
    with tf.compat.v1.name_scope('deserialize_image_record'):
        obj = tf.io.parse_single_example(serialized=record, features=feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label

def get_preprocess_fn(preprocess_method, input_size, mode='validation'):
    """Creates a function to parse and process a TFRecord

    preprocess_method: string
    input_size: int
    mode: string, which mode to use (validation or benchmark)
    returns: function, the preprocessing function for a record
    """
    if preprocess_method == 'vgg':
        preprocess_fn = vgg_preprocessing
    elif preprocess_method == 'inception':
        preprocess_fn = inception_preprocessing
    else:
        raise ValueError(
            'Invalid preprocessing method {}'.format(preprocess_method))

    def validation_process(record):
        # Parse TFRecord
        imgdata, label = deserialize_image_record(record)
        label -= 1  # Change to 0-based (don't use background class)
        try:
            image = tf.image.decode_jpeg(
                imgdata, channels=3, fancy_upscaling=False, dct_method='INTEGER_FAST')
        except:
            image = tf.image.decode_png(imgdata, channels=3)
        # Use model's preprocessing function
        image = preprocess_fn(image, input_size, input_size)
        return image, label

    def benchmark_process(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = preprocess_fn(image, input_size, input_size)
        return image

    if mode == 'validation':
        return validation_process
    if mode == 'benchmark':
        return benchmark_process
    raise ValueError("Mode must be either 'validation' or 'benchmark'")

def input_fn(data_files, batch_size, width, height, labels,
             use_synthetic, preprocess_style, mode='validation',
             return_numpy=False):
    if use_synthetic:
        input_width, input_height = width, height
        features = np.random.normal(
            loc=112, scale=70,
            size=(batch_size, input_height, input_width, 3)).astype(np.float32)
        features = np.clip(features, 0.0, 255.0)
        labels = np.random.choice(
            a=labels,
            size=(batch_size),
            dtype=np.int32)
        if not return_numpy:
            with tf.device('/device:GPU:0'):
                features = tf.convert_to_tensor(tf.get_variable(
                    "features", dtype=tf.float32, initializer=tf.constant(features)))
                labels = tf.identity(tf.constant(labels))
                # ar_dataset = tf.data.Dataset.from_tensor_slices([features])
                # ar_dataset = ar_dataset.repeat()
    else:
        assert not return_numpy, 'return_numpy canot be used when use_synthetic is False'
        # preprocess function for input data
        if preprocess_style == 'vgg':
            preprocess_fn = vgg_preprocessing
        elif preprocess_style == 'inception':
            preprocess_fn = inception_preprocessing

        if mode == 'tfrecords':
            dataset = tf.data.TFRecordDataset(data_files)
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=8))
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            dataset = dataset.repeat(count=1)
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
        elif mode == 'images':
            dataset = tf.data.Dataset.from_tensor_slices(data_files)
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=8))
            dataset = dataset.repeat(count=1)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            labels = tf.identity(tf.constant(labels))
        else:
            raise ValueError("Mode must be either 'tfrecords' or 'images'")
        """
        to do TF 2.0 alternatives
        if mode == 'validation':
            ar_dataset = tf.data.TFRecordDataset(data_files)
            ar_dataset = ar_dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
            ar_dataset = ar_dataset.batch(batch_size=batch_size)
            ar_dataset = ar_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            ar_dataset = ar_dataset.repeat(count=1)
        elif mode == 'benchmark':
            ar_dataset = tf.data.Dataset.from_tensor_slices(data_files)
            ar_dataset = ar_dataset.map(map_func=preprocess_fn, num_parallel_calls=8)
            ar_dataset = ar_dataset.batch(batch_size=batch_size)
            ar_dataset = ar_dataset.repeat(count=1)
        """
    return features, labels


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_files(data_dir, filename_pattern):
    if data_dir == None:
        return []
    files = tf.io.gfile.glob(os.path.join(data_dir, filename_pattern))
    if files == []:
        raise ValueError('Can not find any files in {} with '
                         'pattern "{}"'.format(data_dir, filename_pattern))
    return files


def print_dict(input_dict, str='', scale=None):
    for k, v in sorted(input_dict.items()):
        headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
        v = v * scale if scale else v
        print('{}{}'.format(headline, '%.1f' % v if type(v) == float else v))


def check_tensor_core_gpu_present():
    local_device_protos = device_lib.list_local_devices()
    for line in local_device_protos:
        if "compute capability" in str(line):
            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1])
            if compute_capability >= 7.0:
                return True

def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return
    print('Found the following GPUs:')
    for gpu in gpus:
        print(' ', gpu)
    for gpu in gpus:
        try:
            if not gpu_mem_cap:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=gpu_mem_cap)])
        except RuntimeError as e:
            print('Can not set GPU memory config', e)

def save_trt_engine(graph_func):
    gen_trt_ops = LazyLoader(
        "gen_trt_ops", globals(),
        "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")

    trt_graph = graph_func.graph.as_graph_def()
    for n in trt_graph.node:
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
            with tf.io.gfile.GFile("%s.plan" % (n.name.replace("/", "_")), 'wb') as f:
                f.write(n.attr["serialized_segment"].s)
            print(n.attr["serialized_segment"])
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))
