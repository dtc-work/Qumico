import json
import platform
import os
import subprocess
from os import path


from onnx import helper, save, shape_inference
from onnx import mapping, checker
from qumico.common.tflite_helper import TFLITE_TENSOR_TYPE_TO_NP_TYPE
from qumico.ir import tflite
from qumico import QumicoRootPath

# todo decorator adaptive
import qumico.common.keras_helper as k_helper
import qumico.common.tensorflow_helper as tf_converter
from qumico.common.tflite_helper import DataFormat
from qumico.handlers.frontend import tflite as tflite_ops


class BaseFrontend:
    pass


class KerasFrontend(BaseFrontend):

    @classmethod
    def convert_from_keras(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                           output_op_name, K, binary_flag=True, **kwargs):
        onnx_file = k_helper.keras_save_to_onnx(K=K, output_path=output_path, pb_name=pb_name, ckpt_name=ckpt_name,
                                                cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                onnx_name=onnx_name, model_name=model_name,
                                                output_op_name=output_op_name,
                                                binary_flag=binary_flag)
        return onnx_file

    @classmethod
    def convert_from_yaml(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                          yaml_file, h5_file, binary_flag=True, **kwargs):
        onnx_file = k_helper.load_yamlh5_to_onnx(output_path=output_path, onnx_name=onnx_name, model_name=model_name,
                                                 ckpt_name=ckpt_name, pb_name=pb_name, cache_path=cache_path,
                                                 freeze_pb_name=freeze_pb_name, yaml_file=yaml_file, h5_file=h5_file,
                                                 binary_flag=binary_flag)
        return onnx_file

    @classmethod
    def convert_from_json(cls, output_path, onnx_name, model_name, ckpt_name, pb_name, cache_path, freeze_pb_name,
                          json_file, h5_file, binary_flag=True, **kwargs):
        onnx_file = k_helper.load_jsonh5_to_onnx(output_path=output_path, onnx_name=onnx_name, model_name=model_name,
                                                 ckpt_name=ckpt_name, pb_name=pb_name, cache_path=cache_path,
                                                 freeze_pb_name=freeze_pb_name, json_file=json_file, h5_file=h5_file,
                                                 binary_flag=binary_flag)
        return onnx_file


class TensorflowFrontend(BaseFrontend):

    @classmethod
    def convert_from_tensorflow(cls, output_path, onnx_name, model_name, output_op_name, ckpt_file, pb_file, cache_path, freeze_pb_name,
                                binary_flag=True, **kwargs):
        onnx_file = tf_converter.tensorflow_save_onnx(output_path=output_path, onnx_name=onnx_name,
                                                      model_name=model_name, output_op_name=output_op_name, ckpt_file=ckpt_file, pb_file=pb_file,
                                                      cache_path=cache_path, freeze_pb_name=freeze_pb_name,
                                                      binary_flag=binary_flag)
        return onnx_file


class TFLiteFrontend(BaseFrontend):

    def __init__(self, flatc_path=None, schema_path=None,
                 data_format=DataFormat.channels_first):

        flatbuffers_path = path.join(QumicoRootPath, "..", "external", "flatbuffers")
        if flatc_path is None:
            if platform.system() == "Windows":
                self._flatc_path = path.join(flatbuffers_path, "win", "flatc.exe")
            elif platform.system() == "Linux":
                self._flatc_path = path.join(flatbuffers_path, "flatc")
            else:
                raise RuntimeError()
        else:
            self._flatc_path=flatc_path

        if schema_path is None:
            self._schema_path=path.join(flatbuffers_path, "schema.fbs")
        else:
            self._schema_path = schema_path

        self._model = None
        self._json_path = ""
        self._model_name = ""
        self._data_format = data_format

    @property
    def data_format(self):
        return self._data_format

    @property
    def flatc_path(self):
        return self._flatc_path

    @property
    def schema_path(self):
        return self._schema_path

    @property
    def model_name(self):
        return self._model_name

    def print_model(self):
        self._model.print()

    def _conv_tflite_to_json(self, tflite_path, export_path):

        if not os.access(self.flatc_path, os.R_OK) or not os.access(self.flatc_path, os.X_OK):
            raise IOError(f"flatcのファイルがアクセスが拒否されました。 {self.flatc_path}")

        args = list()
        args.append(path.join(".", self.flatc_path))
        args.append("-o")
        args.append(export_path)
        args.append("--json")
        args.append("--strict-json")
        args.append("--defaults-json")
        args.append(self.schema_path)
        args.append("--")
        args.append(tflite_path)

        p = subprocess.Popen(args=args)
        p.wait()

        json_name = (os.path.splitext(os.path.basename(tflite_path))[0]) + ".json"
        return os.path.join(export_path, json_name)

    def _parse_json(self, json_path):

        with open(json_path) as f:
            net_json: dict = json.load(f)

        self._model = tflite.TFLiteModelv3.parse(net_json)
        self._model_name = path.basename(json_path).replace(".json", "")

    def _create_onnx_tensor(self, nodes):
        onnx_tensors = []
        register_names = []
        for n in nodes:
            for tensor in n.onnx_tensors:
                checker.check_tensor(tensor)
                if not tensor.name in register_names:
                    register_names.append(tensor.name)
                    onnx_tensors.append(tensor)
        return onnx_tensors

    def _create_onnx_graph(self, nodes, name, inputs, outputs, initializer, doc_string, value_info):
        graph = helper.make_graph(nodes=nodes,
                                  name=name,
                                  inputs=inputs,
                                  outputs=outputs,
                                  initializer=initializer,
                                  doc_string=doc_string,
                                  value_info=value_info)
        checker.check_graph(graph)
        return graph

    def _create_onnx_model(self, graph, **kwargs):
        model = helper.make_model(graph, **kwargs)
        checker.check_model(model)
        return model

    def _create_value_info(self, tflite_tensor):

        np_type = TFLITE_TENSOR_TYPE_TO_NP_TYPE[tflite_tensor.tensor_type]
        onnx_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np_type]

        return helper.make_tensor_value_info(name=tflite_tensor.name,
                                             elem_type=onnx_type,
                                             shape=tflite_tensor.shape)

    def _conv_json_to_onnx(self, export_path):
        tflite_model = self._model
        tflite_tensors = self._model.subgraphs.tensors
        operator_codes = self._model.operator_codes

        # graph name
        onnx_graph_name = self._model_name

        # graph input& output
        input_node_index = tflite_model.subgraphs.inputs[0]
        output_node_index = tflite_model.subgraphs.outputs[0]

        onnx_graph_input = self._create_value_info(tflite_tensors[input_node_index])
        onnx_graph_output = self._create_value_info(tflite_tensors[output_node_index])

        nodes = []

        for o in tflite_model.subgraphs.operators:  #

            # todo: op & version adaptive
            op_type_name = operator_codes[o.opcode_index].builtin_code
            # inputs
            tflite_input_tensors = [tflite_tensors[idx] for idx in o.inputs]
            tflite_output_tensors = [tflite_tensors[idx] for idx in o.outputs]

            tflite_input_buffers = [tflite_model.np_buffers[tensor.buffer] for tensor in tflite_input_tensors]
            tflite_output_buffers = [tflite_model.np_buffers[tensor.buffer] for tensor in tflite_output_tensors]

            ops = tflite_ops.clsmembers

            if hasattr(ops[op_type_name], "TFLITE_OP") and ops[op_type_name].TFLITE_OP == op_type_name:
                nodes.append(ops[op_type_name].create_onnx_node(operator=o,
                                                                inputs=tflite_input_tensors,
                                                                outputs=tflite_output_tensors,
                                                                input_buffers=tflite_input_buffers,
                                                                output_buffers=tflite_output_buffers,
                                                                data_format=self.data_format))

            else:
                raise ValueError("OP Not Suppoted")

        # onnx node
        onnx_nodes = []
        for n in nodes:
            for on in n.onnx_nodes:
                checker.check_node(on)
                onnx_nodes.append(on)
                # print(getattr(n, "version_1", None))

        # graph value_info
        graph_value_info = []
        register_names = [onnx_graph_input.name, onnx_graph_output.name]

        for n in nodes:
            for value_info in n.onnx_value_infos:
                if value_info.name in register_names:
                    continue
                register_names.append(value_info.name)
                checker.check_value_info(value_info)
                graph_value_info.append(value_info)

        onnx_tensors = self._create_onnx_tensor(nodes)
        graph = self._create_onnx_graph(nodes=onnx_nodes,
                                        name=onnx_graph_name,
                                        inputs=[onnx_graph_input],
                                        outputs=[onnx_graph_output],
                                        initializer=onnx_tensors,
                                        doc_string=tflite_model.description,
                                        value_info=graph_value_info)

        onnx_model = self._create_onnx_model(graph)

        inferred_model = shape_inference.infer_shapes(onnx_model)
        save(inferred_model, path.join(export_path, '{0}.onnx'.format(self.model_name)))

    def convert_tflite_to_onnx(self, tflite_path, export_path):

        if self.flatc_path is None or not path.exists(self.flatc_path):
            raise FileNotFoundError("flatc pathを見つかりません。")

        if self.schema_path is None or not path.exists(self.schema_path):
            raise FileNotFoundError("schema_path pathを見つかりません。")

        if tflite_path is None or not path.exists(tflite_path):
            raise FileNotFoundError("tfliteファイルを見つかりません。")

        if export_path is None:
            raise FileNotFoundError("export pathを見つかりません。")

        json_file = self._conv_tflite_to_json(tflite_path, export_path)

        self._parse_json(json_file)
        self._conv_json_to_onnx(export_path)
        print(path.join(export_path, self.model_name+".onnx")+"を作成しました。")
