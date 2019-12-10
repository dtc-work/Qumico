from os import path

from onnx import defs, numpy_helper, TensorProto 
from onnx.backend.base import Backend
from onnx.helper import make_tensor_value_info, make_graph, make_model, make_opsetid
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


from qumico.common import exception
from qumico.common import supports_device as common_supports_device
from qumico.backend_rep import QumicoRep
from qumico.common.handler_helper import get_all_backend_handlers
from qumico.common.node import QumicoNode
from qumico.common import value_info_converter
from qumico import QUMICO_EXPORT_ROOT_PATH
import qumico.handlers.optimize as optimizer

class QumicoBackend(Backend):
    """ Qumico Backend for ONNX
    """
    out_c_path = path.abspath(QUMICO_EXPORT_ROOT_PATH)
    def __init__(self, out_c_path=None):

        if out_c_path is not None:
            QumicoBackend.out_c_path = out_c_path

    
    @classmethod
    def _onnx_graph_to_qumico_rep(cls, graph_def, opset, device, optimize, strict):
        """ Convert ONNX graph to QucmioRep.
    
        :param graph_def: ONNX GraphProto object.
        :param opset: ONNX OperatorSetIdProto list.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted tensorflow model.
        :return: QucmioRep object.
        """
        
        # tf_rep_graph = tf.Graph()
        # with tf_rep_graph.as_default():
        # initializer: TensorProtos representing the values to initialize
        # a given tensor.
        # initialized: A list of names of the initialized tensors.
        if graph_def.initializer: 
            input_dict_items = [(QumicoNode.valid_var_name(i.name), numpy_helper.to_array(i)) for i in graph_def.initializer]
            initialized = {init.name for init in graph_def.initializer}
        else:
            input_dict_items = []
            initialized = set()

        # creating placeholders for currently unknown inputs
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue
            x = value_info_converter.conv(value_info)
            input_dict_items.append((QumicoNode.valid_var_name(value_info.name), x))


        # tensor dict: this dictionary is a map from variable names
        # to the latest produced TF tensors of the given name.
        # This dictionary will get updated as we build the graph to
        # record the names of newly produced tensors.
        tensor_dict = dict(input_dict_items)
        # Since tensor dict may be updated, we need to keep a copy
        # of the original input dict where we track the earliest
        # defined tensors so we can have access to the placeholders
        # to feed in input tensors when we run the graph.
        # input_dict = dict(input_dict_items)
        nodes=[]

        o = value_info_converter.conv(graph_def.output[0])
        prev = None

        for node in graph_def.node:
            onnx_node = QumicoNode(node,inputs=tensor_dict, opset=opset, outputs_info=(o.dtype, o.shape), device=device)
            tensor_dict.update(onnx_node.op.output_tensor._asdict())
            if optimize is not None:
                if optimizer.FusePrevTranspose in optimize.options:
                    if prev is not None and optimizer.FusePrevTranspose.validate(prev, onnx_node):                    
                        fuseOp = optimizer.FusePrevTranspose(prev, onnx_node)
                        del nodes[-1]                    
                        onnx_node.conv_optimize_op(fuseOp)
                    
            nodes.append(onnx_node)
            prev= onnx_node

        qumico_rep = QumicoRep(cls.out_c_path)
        qumico_rep.graph = nodes
        qumico_rep.inputs = [
            QumicoNode.valid_var_name(value_info.name)
            for value_info in graph_def.input
            if value_info.name not in initialized
            ]
        qumico_rep.outputs = [QumicoNode.valid_var_name(value_info.name) for value_info in graph_def.output]
        qumico_rep.initializers =[QumicoNode.valid_var_name(i) for i in initialized]
        qumico_rep.tensor_dict = tensor_dict

        return qumico_rep
    
    
    @classmethod
    def prepare(cls, model, device='CPU', optimize=None, strict=True, **kwargs):
        """Prepare an ONNX model for Qumico Backend.
        This function converts an ONNX model to an internel representation
        of the computational graph called QumicoRep and returns
        the converted representation.
        :param model: The ONNX model to be converted.
        :param device: The device to execute this model on.
        :param strict: Whether to enforce semantic equivalence between the original model
          and the converted c-language-based model, defaults to True (yes, enforce semantic equivalence).
          Changing to False is strongly discouraged.
          Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
        :returns: A QumicoRep class object representing the ONNX model
        """

        super(QumicoBackend, cls).prepare(model, device, **kwargs)
        rep = cls.onnx_model_to_qumico_rep(model, device, optimize, strict)
        return rep


    @classmethod
    def onnx_model_to_qumico_rep(cls, model, device,optimize, strict):
        """ Convert ONNX model to QumicoRep.
        :param model: ONNX ModelProto object.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted tensorflow model.
        :return: TensorflowRep object.
        """

        # Models with IR_VERSION less than 3 does not have opset_import set.
        # We default to minimum opset, this behavior is consistent with
        # onnx checker.
        # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
        if model.ir_version < 3:
            opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset_import = model.opset_import
        return cls._onnx_graph_to_qumico_rep(model.graph, opset_import, device,optimize, strict)

    
    @classmethod
    def run_model(cls,
                  model,  # type: ModelProto
                  inputs,  # type: Any
                  device='CPU',  # type: Text
                  **kwargs  # type: Any
                  ):  # type: (...) -> Tuple[Any, ...]
        backend = cls.prepare(model, device, **kwargs)
        
        assert backend is not None
        # arugment check
        if inputs is None: # can convert ndpointer
            inputs = []

        return backend.run(inputs)


    @classmethod
    def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
        """ Run ONNX node.
        :param node: ONNX NodeProto object.
        :param inputs: Inputs.
        :param device: Device run on.
        :param outputs_info: None.
        :param kwargs: Other args.
        :return: Outputs.
        """
        # TODO Remove Optional from return type
        super(QumicoBackend, cls).run_node(node, inputs, device)
        
        if inputs is None: # can convert ndpointer
            inputs = []

        # default values for input/output tensors
        # data_type
        input_tensor_types = [NP_TYPE_TO_TENSOR_TYPE[node_input.dtype] for node_input in inputs]
        output_tensor_types = [TensorProto.FLOAT for idx in range(len(node.output))]
        output_tensor_shapes = [()]  # type: List[Tuple[int, ...]]

        if outputs_info is not None:
            output_tensor_types = [NP_TYPE_TO_TENSOR_TYPE[dtype] for (dtype, shape) in
                                   outputs_info]
            output_tensor_shapes = [shape for (dtype, shape) in outputs_info]

        input_tensors = [make_tensor_value_info(name, tensor_type, value.shape)
                         for name, value, tensor_type in zip(node.input, inputs,
                                                             input_tensor_types)]
        output_tensors = [make_tensor_value_info(name, tensor_type, shape)
                          for name, shape, tensor_type in zip(node.output, output_tensor_shapes,
                                                              output_tensor_types)]
        graph = make_graph([node], 'graph_from_one_node', input_tensors, output_tensors)
        model = make_model(graph, producer_name='QumicoBackend')
        if 'opset_version' in kwargs:
            model.opset_import[0].version = kwargs['opset_version']
        return cls.prepare(model, device,**kwargs).run(inputs, **kwargs)


    @classmethod
    def supports_device(cls, device):
        return common_supports_device(device)


prepare = QumicoBackend.prepare

run_node = QumicoBackend.run_node

run_model = QumicoBackend.run_model

supports_device = QumicoBackend.supports_device
