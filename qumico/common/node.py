from collections import defaultdict, OrderedDict
from inspect import cleandoc

import numpy as np

from onnx.helper import make_opsetid
from onnx import defs
from onnx.backend.base import namedtupledict


from qumico import SUPPORT_ONNX_OPSET
from qumico.common import exception
from qumico import pb_wrapper
from qumico.common.handler_helper import get_all_backend_handlers

class QumicoNode(pb_wrapper.OnnxNode):

    NAME_GEN = defaultdict(int)
    GEN_NODE_PARAM_NAME = 0
    VI_PREFIX = "vi_" # ValueInfo Name Prefix

    def __init__(self, node, inputs, outputs_info=None, device=None):

        super(QumicoNode, self).__init__(node)
        if self._name == "":
            self._name = self.op_type +"Node"+ str(self.NAME_GEN[self.op_type])

        self.NAME_GEN[self.op_type] += 1
        self.__class__.GEN_NODE_PARAM_NAME +=1 # count up

        field_values = OrderedDict()
        for i in map(lambda s:QumicoNode.valid_var_name(s), self.inputs):
            input_val =np.expand_dims(inputs[i],0) if inputs[i].ndim ==0 else inputs[i]
            field_values.update({i: input_val})

        self._input_tensor = namedtupledict("input_tensor", field_values.keys())(**field_values)
        self.outputs_info =outputs_info
        op = self._onnx_node_to_qumico_op(self, tensor_dict=inputs, device=device)
        self._op = op
        self._output_tensor = op.output_tensor
        self._node_param_name = "NodeParam" + str(self.GEN_NODE_PARAM_NAME)

    def conv_optimize_op(self, optimizie_op):
        self._op = optimizie_op
          
    @classmethod
    def valid_var_name(cls, original):
        return cls.VI_PREFIX + original.replace('/', '_').replace('-', '_').replace(":","")
    
    @property
    def op(self):
        return self._op
    
    @property
    def node_param_name(self):
        return self._node_param_name

    @property
    def name(self):
        return self._name
 
    @name.setter
    def name(self, n):
        self._name = n

    @property
    def input_tensor(self):
        return self._input_tensor

    @property
    def input_tensor_names(self):
        return self._input_tensor._fields

    @property
    def input_tensor_dict(self):
        return self._input_tensor._asdict()
    
    @property
    def input_tensor_values(self):
        return list(self._input_tensor._asdict().values())

    @property # Cacheable ?
    def input_tensor_shapes(self):
        return list(v.shape for v in self._input_tensor._asdict().values())

    @property
    def input_tensor_dtypes(self):
        return list(v.dtype for v in self._input_tensor._asdict().values())

    @property
    def input_tensor_ndims(self):
        return list(v.ndim for v in self._input_tensor._asdict().values())

    @property
    def output_tensor(self):
        return self._output_tensor

    @property
    def output_tensor_names(self):
        return self._output_tensor._fields

    @property
    def output_tensor_dict(self):
        return self._output_tensor._asdict()
    
    @property
    def output_tensor_values(self):
        return list(self._output_tensor._asdict().values())

    @property
    def output_tensor_shapes(self):
        return list(v.shape for v in self._input_tensor._asdict().values())

    @property
    def output_tensor_dtypes(self):
        return list(v.dtype for v in self._input_tensor._asdict().values())

    @property
    def output_tensor_ndims(self):
        return list(v.ndim for v in self._input_tensor._asdict().values())

    
    def generate_init_node(self, node_num, indent= 4):
        return self.op.gen_init_func(self, node_num, indent)

    
    def generate_run_func(self, node_num,inputs_names=[], outputs_names=[], indent=4):

        TemplateRunFunc = cleandoc("""
        {indent}{param_type} *param{node_num} = ({param_type}*) Nodes[{node_num}].op_param;
        {indent}{op_func_name}(param{node_num}, {args_inouts});
        """)

        args_inouts = []
        args_inouts.extend(inputs_names)
        args_inouts.extend(outputs_names)

        if not len(inputs_names) == 0: # for input_params 
            args_inouts.append("NULL")

        if not len(outputs_names) == 0: # for output_parmas
            args_inouts.append("NULL")

        
        inputs_names = inputs_names if inputs_names is not None else [] 
        
        mapping = {}
        mapping.update({"param_type" : self.op.get_param_type_name()})
        mapping.update({"node_num": str(node_num)})
        mapping.update({"op_func_name": self.op.func_name})
#         mapping.update({"op_func_name": self.op.get_func_name()})
        mapping.update({"args_inouts": ", ".join(args_inouts)})
        mapping.update({"indent": " " * indent})
        
        return TemplateRunFunc.format(**mapping)
        

    def gen_node_variables(self, node_num, **kwargs):
        return self.op.gen_op_variables(self, node_num, **kwargs)


    @classmethod
    def _onnx_node_to_qumico_op(cls,
                                  node,
                                  tensor_dict,
                                  device,
                                  handlers=None,
                                  opset= [make_opsetid(defs.ONNX_DOMAIN, SUPPORT_ONNX_OPSET)],
                                  strict=True):
        """
        Convert onnx node to tensorflow op.

        Args:
            node: Onnx node object.
            tensor_dict: Tensor dict of graph.
            opset: Opset version of the operator set. Default 0 means using latest version.
            strict: whether to enforce semantic equivalence between the original model
                and the converted tensorflow model, defaults to True (yes, enforce semantic equivalence).
                Changing to False is strongly discouraged.
        Returns:
            Qumico op
    """
        handlers = handlers or cls._get_handlers(opset)
        handler = handlers[node.domain].get(node.op_type, None)

        if handler:
            return handler.handle(node, tensor_dict=tensor_dict,device=device, strict=strict)
        else:
            exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type)

    @classmethod
    def _get_handlers(cls, opset):
        """ Get all backend handlers with opset.
        :param opset: ONNX OperatorSetIdProto list.
        :return: All backend handlers.
        """

        opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])

        return get_all_backend_handlers(opset_dict)