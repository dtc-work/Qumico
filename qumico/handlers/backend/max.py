import string
from inspect import cleandoc
from collections import OrderedDict
from itertools import zip_longest

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type
from .math_mixin import ArithmeticMixin


@onnx_op("Max")
class Max(ArithmeticMixin, BackendHandler):
    
    @classmethod
    def instantiate(cls, node, **kwargs):
        output = None
        for i, (_, v) in enumerate(node.input_tensor_dict.items()):
            if i == 0:
                output = v
                continue
            output = np.add(output, v)
        output_value = {node.valid_var_name(node.outputs[0]): 
                        np.ones(shape=output.shape, dtype=output.dtype)}
        output_tensor =namedtupledict("output_tensor", output_value.keys())(**output_value)

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, **kwargs)
    

    @classmethod
    def get_param_type_name(cls):
        return   "MaxOpParam"


    @classmethod
    def get_c_op_file_name(cls):
        return ["max.c"]


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ["math.h", "float.h"]


    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            """
            typedef struct {
                char* name;
                int input_count;
                int ndim;
                int* shape;
                void *value;
            } MaxOpParam;
            """)


    def _gen_array_element_val(self, ndim, target):
        val = ""
        for _, element_num_x, step in zip_longest(range(ndim), target.shape[::-1],
                                               reversed(string.ascii_lowercase[8:8 + self.output_tensor_ndims[0]])):
            if element_num_x is not None :
                if element_num_x == 1:
                    val  =  "[0]" + val
                else:
                    val  =  "[{0}]".format(step) + val

        return val
        
    
    def generate_c_code(self, **kwargs):
        res =""

        # include header
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"

        # 1
        TemplateArrayAddLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]),indent=0)


        # 2
        input_vals = OrderedDict({k: self._gen_array_element_val(self.output_tensor_ndims[0], v)  for k, v in self.input_tensor_dict.items()})
        output_vals = {self.output_tensor_names[0]: self._gen_array_element_val(self.output_tensor_ndims[0], self.output_tensor_values[0])}


        Conditions = ""
        MaxStatement = ""
        TemplateCondition = cleandoc("""
        {t} m = -DBL_MAX;
        {conditions}
        {indent}{outputVal} = m;
        """)

        TemplateCompare = cleandoc("""
        {indent}if (m < {input}){{
        {indent}    m = {input};
        {indent}}}
        """)
        for k, v in input_vals.items():
            Conditions += TemplateCompare.format(**{"input": k + v,
                                                    "indent": " " * 4  * (self.input_tensor_ndims[0] + 1)})
            Conditions += "\n"
        else:
            mapping_cond ={"t": data_type.np2c(self.input_tensor_dtypes[0])}
            mapping_cond.update({"conditions": Conditions})
            mapping_cond.update({"outputVal": list(output_vals.keys())[0] + list(output_vals.values())[0]})
            mapping_cond.update({"indent": " " * 4 * (self.output_tensor_ndims[0] + 1)})
            MaxStatement += TemplateCondition.format(**mapping_cond)

        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param,{InputsParamSignature}, {OutputsParamSignature}, void *inputs_params, void* outputs_params)
        {{
        {statements}
        }}
        """)
        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})

        input_sigs = []
        for name, value in self.input_tensor_dict.items():
            input_sigs.append(self.gen_param_signature(name, value))
        
        mappingf.update({"InputsParamSignature":",".join(input_sigs)})
        mappingf.update({"OutputsParamSignature": self.gen_param_signature(self.output_tensor_names[0],
                                                                           self.output_tensor_values[0])})

        mappingf.update({"statements": TemplateArrayAddLoop.replace("[statements]", MaxStatement)})
        res += "\n\n"
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_param_signature(self, name, value): #
        mapping ={}
        mapping.update({"type": data_type.np2c(value.dtype)})
        mapping.update({"name": name})
        mapping.update({"dim_bracket": c_helper.generate_dim_bracket(value.shape)})
 
        return "{type} {name}{dim_bracket}".format(**mapping)

    def gen_op_variables(self, node, node_num, **kwargs):
        TemplateVariavbles = cleandoc("""
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            """)
        ndim = self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]

        mapping = {}
        mapping .update({"shape": ",".join(map(str,shape[:ndim]))})
        mapping .update({"node_num": str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc("""
        {indent}// define input & output
        {indent}{node_param_name}.ndim = {ndim};
        {indent}{node_param_name}.shape= OpShapeNode{node_num};
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
        {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        """)

        mapping = {}
        mapping.update({"node_param_name": node.node_param_name})
        mapping.update({"node_num": str(node_num)})
        mapping.update({"add_name": self.get_name()})
        mapping.update({"ndim":str(self.output_tensor_ndims[0])})
        mapping.update({"output_val_name": self.output_tensor_names[0]})
        mapping.update({"indent":" " * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def get_attrs_processor_param(cls):
        return {"default": {"axis": 0}}

    @classmethod
    def version_1(cls, node, **kwargs):
        raise NotImplementedError()

    @classmethod
    def version_6(cls, node, **kwargs):
        raise NotImplementedError()

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
  