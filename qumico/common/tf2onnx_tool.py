import qumico.common.freeze_graph_tool as freezer
import tensorflow as tf
import tf2onnx

from  qumico import SUPPORT_ONNX_OPSET

def freeze_model(pb_file, ckpt_file, freeze_pb_file, output_op_name, white_list='', black_list='',
                 binary_flag=True):
    freezer.freeze_graph(input_graph=pb_file, input_checkpoint=ckpt_file,
                         output_graph=freeze_pb_file,
                         output_node_names=output_op_name,
                         variable_names_whitelist=white_list,
                         variable_names_blacklist=black_list,
                         input_binary=binary_flag)


def pb_to_onnx(freeze_pb_file, onnx_file, model_name, output_op_name):
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(freeze_pb_file, 'rb') as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
        tf.import_graph_def(graph_def, name='')

        if tf2onnx.__version__ == '0.3.1':
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph)
            model_proto = onnx_graph.make_model(model_name, output_names=[output_op_name + ':0'])
        elif tf2onnx.__version__ == '0.3.2':
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, output_names=[output_op_name + ':0'])
            model_proto = onnx_graph.make_model(model_name)
        else:
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, output_names=[output_op_name + ':0'],
                                                         opset=SUPPORT_ONNX_OPSET)
            model_proto = onnx_graph.make_model(model_name)

        with open(onnx_file, 'wb') as f:
            f.write(model_proto.SerializeToString())
            print(onnx_file + 'を作成しました。')


def load_pb_ckpt_to_onnx(pb_file, ckpt_file, freeze_pb_file, onnx_file, model_name, output_op_name, white_list='',
                         black_list='', binary_flag=True):
    freeze_model(pb_file, ckpt_file, freeze_pb_file, output_op_name, white_list=white_list, black_list=black_list,
                 binary_flag=binary_flag)
    pb_to_onnx(freeze_pb_file, onnx_file, model_name, output_op_name)
