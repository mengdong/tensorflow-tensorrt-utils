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

import tensorflow as tf

print(tf.__version__)
from tensorflow.python.client import session
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        required=True,
        help="The location of the protobuf (\'pb\') model to visualize.")
    FLAGS, unparsed = parser.parse_known_args()

    graph = tf.Graph()

    with graph.as_default():
        with session.Session() as sess:
            # First deserialize your frozen graph:
            with tf.io.gfile.GFile(FLAGS.model_dir, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                nodes = [n.name + ' => ' + n.op for n in graph_def.node]
                for node in nodes:
                    print(node)
