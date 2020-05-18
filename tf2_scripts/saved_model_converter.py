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

import argparse
import os
import logging
import time
import pprint
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants
from tf_utils.tftrt_helper import config_gpu_memory, str2bool, get_files
from tf_utils.tftrt_helper import print_dict, get_trt_conversion_params
from tf_utils.tftrt_helper import save_trt_engine, input_fn

def get_dataset(data_files, batch_size, width, height, labels,
             use_synthetic, preprocess_style,
             return_numpy=True):
    if use_synthetic:
        input_width, input_height = width, height
        features = np.random.normal(
            loc=112, scale=70,
            size=(batch_size, input_height, input_width, 3)).astype(np.float32)
        features = np.clip(features, 0.0, 255.0)
        labels = np.random.choice(
            a=labels,
            size=(batch_size))
        if not return_numpy:
            with tf.device('/device:GPU:0'):
                features = tf.convert_to_tensor(tf.get_variable(
                    "features", dtype=tf.float32, initializer=tf.constant(features)))
                dataset = tf.data.Dataset.from_tensor_slices([features])
                dataset = dataset.repeat()
            return dataset
        else:
            return features, labels

def get_func_from_saved_model(saved_model_dir, engine_path=False):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    print(graph_func.structured_outputs)
    if engine_path:
        graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
        save_trt_engine(graph_func)
    return graph_func

def get_graph_func(input_saved_model_dir,
                   output_saved_model_dir=None,
                   preprocess_method='vgg',
                   conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
                   use_trt=False,
                   calib_files=None,
                   batch_size=None,
                   use_synthetic=False,
                   optimize_offline=False,
                   engine_path=None):
    """Retreives a frozen SavedModel and applies TF-TRT
    use_trt: bool, if true use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: TF function that is ready to run for inference
    """
    start_time = time.time()
    if use_trt:
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params,
        )

        def input_fn(input_files, num_iterations):
            dataset = get_dataset(data_files=None,
                                  batch_size=batch_size,
                                  width=224,
                                  height=224,
                                  labels=10,
                                  use_synthetic=True,
                                  preprocess_style=preprocess_method)
            for i, (batch_images, _) in enumerate(dataset):
                if i >= num_iterations:
                    break
                yield (batch_images,)
                print("  step %d/%d" % (i + 1, num_iterations))
                i += 1

        if conversion_params.precision_mode != 'INT8':
            print('Graph conversion...')
            converter.convert()
            if optimize_offline:
                print('Building TensorRT engines...')
                converter.build(input_fn=partial(input_fn, data_files, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
        else:
            print('Graph conversion and INT8 calibration...')
            converter.convert(calibration_input_fn=partial(
                input_fn, calib_files, num_calib_inputs // batch_size))
            if optimize_offline:
                print('Building TensorRT engines...')
                converter.build(input_fn=partial(input_fn, data_files, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
    return graph_func, {'conversion': time.time() - start_time}


def eval_fn(preds, labels, adjust):
    """Measures number of correct predicted labels in a batch.
       Assumes preds and labels are numpy arrays.
    """
    preds = preds - adjust
    return np.sum((labels.reshape(-1) == preds).astype(np.float32))

def run_inference(graph_func,
                  data_files,
                  batch_size,
                  preprocess_method,
                  num_classes,
                  num_iterations,
                  num_warmup_iterations,
                  use_synthetic,
                  display_every=100,
                  mode='benchmark',
                  target_duration=None):
    """Run the given graph_func on the data files provided. In validation mode,
    it consumes TFRecords with labels and reports accuracy. In benchmark mode, it
    times inference on real data (.jpgs).
    """
    results = {}
    corrects = 0
    iter_times = []
    adjust = 1 if num_classes == 1001 else 0
    initial_time = time.time()
    dataset = get_dataset(data_files=data_files,
                          batch_size=batch_size,
                          width=224,
                          height=224,
                          labels=10,
                          use_synthetic=True,
                          preprocess_style=preprocess_method)

    for i, batch_images in enumerate(dataset):
        if i >= num_warmup_iterations:
            start_time = time.time()
            batch_preds = list(graph_func(batch_images).values())[0].numpy()
            iter_times.append(time.time() - start_time)
            if i % display_every == 0:
                print("  step %d/%d, iter_time(ms)=%.0f" %
                      (i + 1, num_iterations, iter_times[-1] * 1000))
        else:
            batch_preds = list(graph_func(batch_images).values())[0].numpy()
        if i > 0 and target_duration is not None and \
            time.time() - initial_time > target_duration:
            break
        if num_iterations is not None and i >= num_iterations:
            break

    if not iter_times:
        return results
    iter_times = np.array(iter_times)
    iter_times = iter_times[num_warmup_iterations:]
    results['total_time'] = np.sum(iter_times)
    results['images_per_sec'] = np.mean(batch_size / iter_times)
    results['99th_percentile'] = np.percentile(
        iter_times, q=99, interpolation='lower') * 1000
    results['latency_mean'] = np.mean(iter_times) * 1000
    results['latency_median'] = np.median(iter_times) * 1000
    results['latency_min'] = np.min(iter_times) * 1000
    return results

if __name__ == '__main__':
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='arguments for converting and evaluating TFTRT saved models')

    # TRT conversion params
    parser.add_argument('--use_trt', type=str2bool, default=True,
                        help='If set to True, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--input_saved_model_dir', type=str, default=None,
                        help='Directory containing the input saved model.')
    parser.add_argument('--output_saved_model_dir', type=str, default=None,
                        help='Directory in which the converted model is saved')
    parser.add_argument('--engine_path', type=str, default=None,
                        help='Directory where to write standalone trt engines. '
                             'Engines are written only if the directory '
                             'is provided. This option is ignored when not using tf_trt.')
    parser.add_argument('--use_trt_dynamic_op', type=str2bool, default=True,
                        help='If set to True, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str,
                        choices=['FP32', 'FP16', 'INT8'], default='FP32',
                        help='Precision mode to use. FP16 and INT8 only'
                             'work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of images per batch.')
    parser.add_argument('--calibration_batch_size', type=int, default=32,
                        help='batch size for calibration')
    parser.add_argument('--num_calibration_batches', type=int, default=2,
                        help='Number of batches of images used for calibration ')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
                        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--max_workspace_size', type=int, default=(6 << 30),
                        help='workspace size in bytes')
    parser.add_argument('--use_synthetic', type=str2bool, default=True,
                        help='If set, one batch of random data is'
                             'generated and used at every iteration.')
    parser.add_argument('--optimize_offline', type=bool, default=False,
                        help='If set, TensorRT engines are built'
                             'before runtime.')

    # run inference params

    parser.add_argument('--input_size', type=int, default=224,
                        help='Size of input images expected by the model')
    parser.add_argument('--num_classes', type=int, default=1001,
                        help='Number of classes used when training the model')
    parser.add_argument('--preprocess_style', type=str,
                        choices=['vgg', 'inception'], default='vgg',
                        help='The image preprocessing method')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Directory containing validation data files.')
    parser.add_argument('--calibration_data_path', type=str,
                        help='Directory containing data for calibrating INT8.')
    parser.add_argument('--num_iterations', type=int, default=50,
                        help='How many iterations(batches) to evaluate.'
                             'If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
                        help='Number of iterations executed between'
                             'two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=10,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--input_data_format', type=str.lower, choices=['tfrecords', 'images'], default='images',
                        help='Which input data format to use, tfrecords or images')
    parser.add_argument('--task', type=str.lower, choices=['imageclassification', 'objectdetection'], default='images',
                        help='which task to perform on')
    parser.add_argument('--target_duration', type=int, default=None,
                        help='If set, script will run for specified number of seconds.')
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
                        help='Upper bound for GPU memory in MB.'
                             'Default is 0 which means allow_growth will be used.')
    parser.add_argument('--mode', choices=['validation', 'benchmark'],
                        default='benchmark',
                        help='Which mode to use (validation or benchmark)')
    args = parser.parse_args()

    if (args.precision == 'INT8' and not args.calib_data_dir
        and not args.use_synthetic):
        raise ValueError('--calib_data_dir is required for INT8 mode')
    if (args.num_iterations is not None
        and args.num_iterations <= args.num_warmup_iterations):
        raise ValueError(
            '--num_iterations must be larger than --num_warmup_iterations '
            '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))

    if args.eval_data_path is None and not args.use_synthetic:
        raise ValueError("--data_dir required if you are not using synthetic data")
    if args.use_synthetic and args.num_iterations is None:
        raise ValueError("--num_iterations is required for --use_synthetic")
    if args.use_trt and not args.output_saved_model_dir:
        raise ValueError("--output_saved_model_dir must be set if use_trt=True")

    calibration_files = []
    data_files = []
    if not args.use_synthetic:
        eval_files = get_files(args.eval_data_path, '*')
        calibration_files = get_files(args.calibration_data_path, '*')

    if args.precision == 'FP16':
        precision = trt.TrtPrecisionMode.FP16
    elif args.precision == 'FP32':
        precision = trt.TrtPrecisionMode.FP32
    else:
        precision = trt.TrtPrecisionMode.INT8

    config_gpu_memory(args.gpu_mem_cap)
    params = get_trt_conversion_params(
        args.max_workspace_size,
        precision,
        args.minimum_segment_size,
        args.batch_size,
        args.use_trt_dynamic_op,)

    graph_func, times = get_graph_func(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        preprocess_method=args.preprocess_style,
        conversion_params=params,
        use_trt=args.use_trt,
        calib_files=calibration_files,
        batch_size=args.batch_size,
        use_synthetic=args.use_synthetic,
        optimize_offline=args.optimize_offline,
        engine_path=args.engine_path)


    def print_dict(input_dict, prefix='  ', postfix=''):
        for k, v in sorted(input_dict.items()):
            print('{}{}: {}{}'.format(prefix, k, '%.1f' % v if isinstance(v, float) else v, postfix))


    print('Benchmark arguments:')
    print_dict(vars(args))
    print('TensorRT Conversion Params:')
    print_dict(dict(params._asdict()))
    print('Conversion times:')
    print_dict(times, postfix='s')

    results = run_inference(graph_func,
                            data_files=data_files,
                            batch_size=args.batch_size,
                            preprocess_method=args.preprocess_style,
                            num_classes=args.num_classes,
                            num_iterations=args.num_iterations,
                            num_warmup_iterations=args.num_warmup_iterations,
                            use_synthetic=args.use_synthetic,
                            display_every=args.display_every,
                            mode=args.mode,
                            target_duration=args.target_duration)

    if args.mode == 'validation':
        print('  accuracy: %.2f' % (results['accuracy'] * 100))
    print('  images/sec: %d' % results['images_per_sec'])
    print('  99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('  total_time(s): %.1f' % results['total_time'])
    print('  latency_mean(ms): %.2f' % results['latency_mean'])
    print('  latency_median(ms): %.2f' % results['latency_median'])
    print('  latency_min(ms): %.2f' % results['latency_min'])
