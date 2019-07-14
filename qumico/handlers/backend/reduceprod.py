import string
from inspect import cleandoc

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type
from .math_mixin import ReductionMixin


@onnx_op("ReduceProd")
class ReduceProd(ReductionMixin, BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):

        input = node.input_tensor[0]

        axes = tuple(node.attrs.get("axes", list(range(input.ndim))))
        keepdims = node.attrs.get("keepdims", True)

        output = np.prod(input, axis=axes, keepdims=keepdims)
        output_value ={node.valid_var_name(node.outputs[0]): output}
        output_tensor = namedtupledict("output_tensor", output_value.keys())(**output_value)

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, attrs=node.attrs,
                   axes=axes, keepdims=keepdims)
    
    @classmethod
    def get_param_type_name(cls):
        return   "ReduceProdOpParam"    

    @classmethod
    def get_c_op_file_name(cls):
        return ["reduceprod.c"]

    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return []
    
    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            """
            typedef struct {
                char* name;
                int ndim;
                int* shape;
                void *value;
            } ReduceProdOpParam;
            """)

    def generate_c_code_init_output(self):
        output = self.output_tensor[0]

        alpha_iter = reversed(string.ascii_lowercase[8:8 + output.ndim])
        indent = [" " * 4] * output.ndim
        TemplateArrayLoop = "".join(indent) + "[statements]"
        output_dim = ""

        for d in reversed(range(output.ndim)):
            var = next(alpha_iter)
            params = {}
            params.update({"var": var})
            params.update({"start": str(0)})
            params.update({"end": str(output.shape[d])})
            
            output_dim ="[" + var + "]" + output_dim
            loop_start ="".join(indent) + "for(int {var}={start};{var}<{end};{var}++ ){{".format(**params ) 
            loop_end = "".join(indent) +"}"

            TemplateArrayLoop =loop_start  +"\n" +  TemplateArrayLoop + "\n" + loop_end
            indent.pop()            
        else:
            if output_dim =="":
                output_dim ="[0]"

        return TemplateArrayLoop.replace("[statements]",  "".join(indent) + "output" + output_dim + "=1;")


    def generate_c_code_reduce(self):
        input = self.input_tensor[0]
        indent = [" " * 4] * input.ndim
        
        axes = tuple(self.attrs.get("axes", list(range(input.ndim))))
        keepdims = self.attrs.get("keepdims", True)
        
        indent = [" " * 4] * input.ndim
        alpha_iter = reversed(string.ascii_lowercase[8:8 + input.ndim])

        TemplateArrayLoop = "".join(indent) + "[statements]"
        input_dim = ""
        output_dim = ""
        
        for d in reversed(range(input.ndim)):
            var = next(alpha_iter)
            params = {}
            params.update({"var": var})
            params.update({"start": str(0)})
            params.update({"end": str(input.shape[d])})

            input_dim = "[" + var + "]" + input_dim
                       
            if d in axes:  # to reduce dimension
                if keepdims == True:
                    output_dim ="[0]" + output_dim                    
                else:
                    pass                    
            else: # not to reduce dimension
                output_dim ="[" + var + "]" + output_dim

            loop_start ="".join(indent) + "for(int {var}={start};{var}<{end};{var}++ ){{".format(**params ) 
            loop_end = "".join(indent) +"}"

            TemplateArrayLoop =loop_start  +"\n" +  TemplateArrayLoop + "\n" + loop_end

            indent.pop()

        else:
            if output_dim == "": # when reduce all dimensions
                output_dim ="[0]"
        
        return TemplateArrayLoop.replace("[statements]",  "".join(indent) + "output" + output_dim + "*=" + "input" + input_dim + ";")


    def generate_c_code(self, **kwargs):

        res =""

        # include header
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"
        
        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param,{t} input{XDims}, {t} output{CDims}, void *inputs_params, void* outputs_params)
        {{
        {init_statements}\n
        {main_statements}
        }}
        """)

        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})
        mappingf.update({"XDims":c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({"CDims": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({"main_statements":self.generate_c_code_reduce()})
        mappingf.update({"init_statements":self.generate_c_code_init_output()})
        res += "\n\n"
        res += TemplateFunction.format(**mappingf)

        return res


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
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)