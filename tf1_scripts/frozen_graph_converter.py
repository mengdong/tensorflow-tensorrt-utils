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
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tf_utils.tftrt_helper import input_fn, str2bool, get_files, print_dict
from tensorflow.python.client import session


def run_inference(task, frozen_graph, batch_size, input_tensors, input_shapes,
                  output_nodes, use_synthetic,
                  num_iterations, num_warmup_iterations, display_every=100):
    """Evaluates a frozen graph

    This function evaluates a graph based on the task
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.
    """
    # load frozen graph from file if string, otherwise must be GraphDef
    if use_synthetic:
        if isinstance(frozen_graph, str):
            frozen_graph_path = frozen_graph
            frozen_graph = graph_pb2.GraphDef()
            with open(frozen_graph_path, 'rb') as f:
                frozen_graph.ParseFromString(f.read())
        elif not isinstance(frozen_graph, graph_pb2.GraphDef):
            raise TypeError('Expected frozen_graph to be GraphDef or str')

        # results = {}
        # task = ''.join(task.lower.split())
        tensor_dict = {}
        for tensor, shape in zip(input_tensors, input_shapes):
            tensor_dict[tensor] = np.ones([int(i) for i in shape.split(',')])

        graph = tf.Graph()
        with graph.as_default():
            with session.Session() as sess:
                output_node = tf.import_graph_def(
                    frozen_graph,
                    return_elements=output_nodes)
                for tensor in input_tensors:
                    print(tf.shape(graph.get_tensor_by_name(tensor)))

                for index in range((num_warmup_iterations)):
                    start = time.time()
                    sess.run(output_node, feed_dict=tensor_dict)
                    stop = time.time()
                    print("Time per warm-up run for Tensorflow Inference %d: %f ms" % (index, (stop - start) * 1000))
                times = []
                for index in range((num_iterations)):
                    start = time.time()
                    sess.run(output_node, feed_dict=tensor_dict)
                    stop = time.time()
                    times.append(stop - start)
                    print("Time per run for Tensorflow Inference %d: %f ms" % (index, (stop - start) * 1000))
        print(sum(times) / num_iterations * 1000);


def build_graph(use_trt,
                graph_path=None,
                output_nodes=['logits', 'classes'],
                engine_path=None,
                use_dynamic_op=False,
                precision='FP16',
                batch_size=16,
                calibration_batch_size=32,
                num_calibration_batches=2,
                minimum_segment_size=2,
                max_workspace_size=(3 << 32),
                calibration_files=None,
                use_synthetic=True,
                output_graph_path='trt.graphdef',
                preprocess_style='vgg'):
    """Builds an image classification model by name

    This function take a frozen graph from given model and
    performs some graph processing to produce a graph that is
    well optimized by the TensorRT package in TensorFlow 1.7+.

    model: string, the model path
    use_trt: bool, if true, use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: tensorflow.GraphDef, the TensorRT compatible frozen graph
    """
    num_nodes = {}
    times = {}
    graph_sizes = {}
    print('Loading cached frozen graph from \'%s\'' % graph_path)
    start_time = time.time()
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        frozen_graph = graph_pb2.GraphDef()
        frozen_graph.ParseFromString(f.read())
    times['loading_frozen_graph'] = time.time() - start_time
    num_nodes['input_frozen_graph'] = len(frozen_graph.node)
    graph_sizes['input_frozen_graph'] = len(frozen_graph.SerializeToString())
    # Convert to TensorRT graph
    if use_trt:
        start_time = time.time()
        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=output_nodes,
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision.upper(),
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op
        )
        frozen_graph = converter.convert()
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_frozen_graph'] = len(frozen_graph.node)
        num_nodes['trt_only_node'] = len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp'])
        graph_sizes['trt_frozen_graph'] = len(frozen_graph.SerializeToString())

        if precision == 'INT8':
            num_calibration_files = len(calibration_files)
            print('There are %d calibration files. \n%s\n%s\n...' % (
                num_calibration_files, calibration_files[0], calibration_files[-1]))

            def calibration_input_fn():
                features, _ = input_fn(calibration_files, batch_size,
                                       width=224, height=224, labels=None,
                                       use_synthetic=use_synthetic,
                                       preprocess_style=preprocess_style,
                                       return_numpy=(precision == 'INT8') and use_synthetic)
                return {'input:0': features}

            # INT8 calibration step
            print('Calibrating INT8...')
            start_time = time.time()
            frozen_graph = converter.calibrate(
                fetch_names=output_nodes,
                num_runs=num_calibration_files // calibration_batch_size,
                input_map_fn=calibration_input_fn)
            times['trt_calibration'] = time.time() - start_time
            graph_sizes['trt_after_calibration'] = len(frozen_graph.SerializeToString())
            print('INT8 graph created.')

        if engine_path:
            segment_number = 0
            for node in frozen_graph.node:
                if node.op == "TRTEngineOp":
                    engine = node.attr["serialized_segment"].s
                    export_engine_path = engine_path + '/{}_{}_{}_segment{}.trtengine'.format('trtengine', precision,
                                                                                              batch_size,
                                                                                              segment_number)
                    segment_number += 1
                    with open(export_engine_path, "wb") as f:
                        f.write(engine)

        start_time = time.time()
        with tf.io.gfile.GFile(output_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        times['saving_frozen_graph'] = time.time() - start_time

    return frozen_graph, num_nodes, times, graph_sizes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for converting and evaluating TFTRT frozen graph models, '
                                                 'converting of work for tf1, benchmark work for tf1/2')

    # TRT conversion params
    parser.add_argument('--use_trt', type=str2bool, default=True,
                        help='If set to True, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--graph_path', type=str, default=None,
                        help='original frozen graph path')
    parser.add_argument('--output_nodes', type=str, default='["logits", "classes"]',
                        help='output nodes in a python list repesentation, like ["logits", "classes"]')
    parser.add_argument('--engine_path', type=str, default=None,
                        help='Directory where to write trt engines. Engines are written only if the directory ' \
                             'is provided. This option is ignored when not using tf_trt.')
    parser.add_argument('--use_trt_dynamic_op', type=str2bool, default=True,
                        help='If set to True, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str.upper, choices=['FP32', 'FP16', 'INT8'], default='FP32',
                        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of images per batch.')
    parser.add_argument('--calibration_batch_size', type=int, default=32,
                        help='batch size for calibration')
    parser.add_argument('--num_calibration_batches', type=int, default=2,
                        help='Number of batches of images used for calibration ')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
                        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--max_workspace_size', type=int, default=(1 << 32),
                        help='workspace size in bytes')
    parser.add_argument('--use_synthetic', type=str2bool, default=True,
                        help='If set to True, synthetic data will used for evaluation and calibration')
    parser.add_argument('--output_graph_path', type=str, default='trt.graphdef',
                        help='name of frozen graphs will be saved to disk after conversion.')

    # inference params
    parser.add_argument('--input_tensors', type=str, default='["import/image_tensor:0"]',
                        help='input tensors in a python list repesentation, like ["import/image_tensor:0"]')
    parser.add_argument('--input_shapes', type=str, default='["16,224,224,3"]',
                        help='according input tensors shapes in a python list repesentation, like ["16,224,224,3"]')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Directory containing validation data files.')
    parser.add_argument('--calibration_data_path', type=str,
                        help='Directory containing data for calibrating INT8.')
    parser.add_argument('--num_iterations', type=int, default=50,
                        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
                        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=10,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--input_data_format', type=str.lower, choices=['tfrecords', 'images'], default='images',
                        help='Which input data format to use, tfrecords or images')
    parser.add_argument('--task', type=str.lower, choices=['imageclassification', 'objectdetection'], default='images',
                        help='which task to perform on')
    parser.add_argument('--target_duration', type=int, default=None,
                        help='If set, script will run for specified number of seconds.')
    parser.add_argument('--preprocess_style', type=str, default='vgg',
                        help='If set, script will run for specified number of seconds.')
    parser.add_argument('--gpu_mem_cap', type=int, default=0,
                        help='Upper bound for GPU memory in MB.'
                             'Default is 0 which means allow_growth will be used.')
    args = parser.parse_args()

    if args.precision == 'INT8' and not args.calibration_data_path:
        raise ValueError('--calib_data_dir is required for INT8 mode')
    if args.num_iterations is not None and args.num_iterations <= args.num_warmup_iterations:
        raise ValueError('--num_iterations must be larger than --num_warmup_iterations '
                         '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))

    calibration_files = []
    data_files = []
    if not args.use_synthetic:
        eval_files = get_files(args.eval_data_path, '*')
        calibration_files = get_files(args.calibration_data_path, '*')
    output_nodes = json.loads(args.output_nodes)
    input_tensors = json.loads(args.input_tensors)
    input_shapes = json.loads(args.input_shapes)

    frozen_graph, num_nodes, times, graph_sizes = build_graph(
        use_trt=args.use_trt,
        graph_path=args.graph_path,
        output_nodes=output_nodes,
        engine_path=args.engine_path,
        use_dynamic_op=args.use_trt_dynamic_op,
        precision=args.precision,
        batch_size=args.batch_size,
        calibration_batch_size=args.calibration_batch_size,
        num_calibration_batches=args.num_calibration_batches,
        minimum_segment_size=args.minimum_segment_size,
        max_workspace_size=args.max_workspace_size,
        calibration_files=calibration_files,
        use_synthetic=args.use_synthetic,
        output_graph_path=args.output_graph_path,
        preprocess_style=args.preprocess_style)

    print_dict(vars(args))
    print_dict(num_nodes, str='num_nodes')
    print_dict(graph_sizes, str='graph_size(MB)', scale=1. / (1 << 20))
    print_dict(times, str='time(s)')

    # Evaluate model
    print('running inference...')
    results = run_inference(
        task=args.task,
        frozen_graph=frozen_graph,
        batch_size=args.batch_size,
        input_tensors=input_tensors,
        input_shapes=input_shapes,
        output_nodes=output_nodes,
        use_synthetic=args.use_synthetic,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        display_every=args.display_every)

    """
    # Display results
    print('results of {}:'.format(args.model))
    if args.mode == 'validation':
        print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    images/sec: %d' % results['images_per_sec'])
    print('    99th_percentile(ms): %.2f' % results['99th_percentile'])
    print('    total_time(s): %.1f' % results['total_time'])
    print('    latency_mean(ms): %.2f' % results['latency_mean'])
    print('    latency_median(ms): %.2f' % results['latency_median'])
    print('    latency_min(ms): %.2f' % results['latency_min'])
    """
