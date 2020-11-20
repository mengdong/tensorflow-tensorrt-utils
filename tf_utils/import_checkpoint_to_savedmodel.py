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
import tensorflow as tf
from tensorflow.python.client import session

def checkpoint_to_savedmodel(checkpoint_path, savedmodel_path):
    export_dir = os.path.join(savedmodel_path, '0') # IMPORTANT: each model folder must be named '0', '1', ... Otherwise it will
    loaded_graph = tf.Graph()
    with session.Session(graph=loaded_graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(checkpoint_path + '.meta')
        loader.restore(sess, checkpoint_path)

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING],
                                             strip_default_attrs=True)
    builder.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--checkpoint_dir_prefix",
        type=str,
        default="",
        required=True,
        help="The location of the checkpoint + meta's prefix")
    parser.add_argument(
        "--savedmodel_dir",
        type=str,
        default="",
        required=True,
        help="The location of the savedmodel")
    FLAGS, unparsed = parser.parse_known_args()
    checkpoint_to_savedmodel(FLAGS.checkpoint_dir_prefix, FLAGS.savedmodel_dir)
