# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import saver as saver_lib

FLAGS = None


def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 output_graph,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_whitelist='',
                                 variable_names_blacklist='',
                                 input_meta_graph_def=None,
                                 input_saved_model_dir=None,
                                 saved_model_tags=None):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if (not input_saved_model_dir and
            not saver_lib.checkpoint_exists(input_checkpoint)):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        if input_meta_graph_def:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ''
        elif input_graph_def:
            for node in input_graph_def.node:
                node.device = ''

    if input_graph_def:
        _ = importer.import_graph_def(input_graph_def, name='')
    with session.Session() as sess:
        if input_saver_def:
            saver = saver_lib.Saver(saver_def=input_saver_def)
            saver.restore(sess, input_checkpoint)
        elif input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(
                input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.split(','))
        elif input_saved_model_dir:
            if saved_model_tags is None:
                saved_model_tags = []
            loader.load(sess, saved_model_tags, input_saved_model_dir)
        else:
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ':0')
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor
            saver = saver_lib.Saver(var_list=var_list)
            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.split(','))

        variable_names_whitelist = (variable_names_whitelist.split(',')
                                    if variable_names_whitelist else None)
        variable_names_blacklist = (variable_names_blacklist.split(',')
                                    if variable_names_blacklist else None)

        if input_meta_graph_def:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_meta_graph_def.graph_def,
                output_node_names.split(','),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)
        else:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)

    # Write GraphDef to file if output path has been given.
    if output_graph:
        with gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def


def _parse_input_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into GraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = 'rb' if input_binary else 'r'
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    return input_graph_def


def _parse_input_meta_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into MetaGraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input meta graph file '" + input_graph + "' does not exist!")
        return -1
    input_meta_graph_def = MetaGraphDef()
    mode = 'rb' if input_binary else 'r'
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_meta_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_meta_graph_def)
    print("Loaded meta graph file '" + input_graph)
    return input_meta_graph_def


def _parse_input_saver_proto(input_saver, input_binary):
    """Parser input tensorflow Saver into SaverDef proto."""
    if not gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1
    mode = 'rb' if input_binary else 'r'
    with gfile.FastGFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
            saver_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), saver_def)
    return saver_def


def freeze_graph(input_graph='',
                 input_checkpoint='',
                 output_graph='',
                 output_node_names='',
                 input_saver='',
                 input_binary=True,
                 restore_op_name='save/restore_all',
                 filename_tensor_name='save/Const:0',
                 clear_devices=True,
                 initializer_nodes='',
                 variable_names_whitelist='',
                 variable_names_blacklist='',
                 input_meta_graph=None,
                 input_saved_model_dir=None,
                 saved_model_tags=tag_constants.SERVING):
    """Converts all variables in a graph and checkpoint into constants."""
    input_graph_def = None
    if input_saved_model_dir:
        input_graph_def = saved_model_utils.get_meta_graph_def(
            input_saved_model_dir, saved_model_tags).graph_def
    elif input_graph:
        input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
    input_meta_graph_def = None
    if input_meta_graph:
        input_meta_graph_def = _parse_input_meta_graph_proto(
            input_meta_graph, input_binary)
    input_saver_def = None
    if input_saver:
        input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
    freeze_graph_with_def_protos(
        input_graph_def, input_saver_def, input_checkpoint, output_node_names,
        restore_op_name, filename_tensor_name, output_graph, clear_devices,
        initializer_nodes, variable_names_whitelist, variable_names_blacklist,
        input_meta_graph_def, input_saved_model_dir, saved_model_tags.split(','))
