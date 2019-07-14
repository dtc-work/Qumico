import string
from inspect import cleandoc
from collections import OrderedDict

import numpy as np


from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op("Softmax")

class Softmax(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data = node.input_tensor[0]
        if (node.attrs.get("axis") is None):
            node.attrs.update({"axis": 1})
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=input_data.shape, dtype=input_data.dtype)}
        output_tensor = namedtupledict("output_tensor", outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return "SoftmaxOpParam"


    @classmethod
    def get_c_op_file_name(cls):
        return ["softmax.c"]


    @classmethod
    def get_c_op_include_header(cls):
        return ["math.h"]
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            """
            typedef struct {
                char* name;
            } SoftmaxOpParam;
            """)


    def generate_c_code(self, **kwargs):
        res =""
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"
    

        axis = self.attrs["axis"]
        batch_size = 1
        for d in range(0, axis):
            batch_size *= self.input_tensor_shapes[0][d]
        num = 1
        for d in range(axis, self.input_tensor_ndims[0]):
            num *= self.input_tensor_shapes[0][d] 
    
        TemplateStatements = """
            {t}   *_input = ({t} *)input;
            {t}   *_output = ({t} *)output;
            int    batch_size = {batch_size};
            int    num = {num};

            int    i;
            int    batch;
            {t}  max, sum;

            for (batch=0; batch<batch_size; batch++) {{
                sum = 0.0;
                max = -HUGE_VAL;
                for (i=0; i<num; i++) {{
                    if (*(_input + batch*num +i) > max) {{
                        max = *(_input + batch*num +i);
                    }}
                }}
                for (i=0; i<num; i++) {{
                    *(_output + batch*num +i) = {exp}(*(_input + batch*num +i) - max);
                    sum += *(_output + batch*num +i);
                }}
                for (i=0; i<num; i++) {{
                    *(_output + batch*num +i) /= sum;
                }}
            }}
        """

        mapping = {}
        mapping.update({"batch_size": batch_size})
        mapping.update({"num": num})
        mapping.update({"d1": self.output_tensor_shapes[0][0]})
        mapping.update({"d2": self.output_tensor_shapes[0][1]})
        mapping.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
        if (self.output_tensor_dtypes[0] == 'float64'):
            mapping.update({"exp": "exp"})
        elif (self.output_tensor_dtypes[0] == 'float32'):
            mapping.update({"exp": "expf"})
        else:
            mapping.update({"exp": "expf"})

        # 3        
        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param, {t} input{dims_input}, {t} output{dims}, void *inputs_params, void* outputs_params)
        {{
            {statements}
        }}
        """)

        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})
        mappingf.update({"input": self.input_tensor_names[0]})
        mappingf.update({"dims_input": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({"output": self.output_tensor_names[0]})
        mappingf.update({"dims": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({"statements": TemplateStatements.format(**mapping)})
        res += "\n\n"
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        TemplateVariavbles = cleandoc("""
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            """)
        ndim =  self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]

        mapping = {}
        mapping.update({"shape": ",".join(map(str,shape[:ndim]))})
        mapping.update({"node_num": str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc("""
        {indent}// define input & output
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
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


