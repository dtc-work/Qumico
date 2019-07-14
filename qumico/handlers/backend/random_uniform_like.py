from inspect import cleandoc
import string

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op("RandomUniformLike")
class RandomUniformLike(BackendHandler):


    @classmethod
    def instantiate(cls, node, **kwargs):

        if node.attrs.get("seed") is not None:
            raise ValueError("seed is not supported")

        A = node.input_tensor_values[0]

        output_dtype =node.attrs.get("dtype", A.dtype)
        output_value = {node.valid_var_name(node.outputs[0]): 
                        np.ones(shape=A.shape, dtype=output_dtype)}
        output_tensor =namedtupledict("output_tensor", output_value.keys())(**output_value)
        
        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, attrs=node.attrs,
                    **kwargs)
    

    @classmethod
    def get_param_type_name(cls):
        return   "RandomUniformLikeOpParam"

    @classmethod
    def get_c_op_file_name(cls):
        return ["randomuniformlike.c"]   
    
    
    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ["stdlib.h"]
    


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
            } RandomUniformLikeOpParam;
            """)
    

    def generate_c_code(self, **kwargs):
        res =""
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"

        # 1
        TemplateArrayDropoutLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]))


        TemplateStatements = """
                    output{dims} = ((high - low) * rand()/RAND_MAX ) - low;
        """

        mapping = {}
        mapping.update({"dims": "".join(["[" + v + "]"  for v in  string.ascii_lowercase[8:8+self.output_tensor_ndims[0]]])}) 

        # 3        
        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param, {t_in} data{dims}, {t_out} output{dims}, void *inputs_params, void* outputs_params) {{
            const float high = {high};
            const float low = {low};
            {statements}
        }}
        """)

        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})
        mappingf.update({"dims": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({"t_in": data_type.np2c(self.input_tensor_dtypes[0])})
        mappingf.update({"t_out": data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({"high": self.attrs.get("high" ,1.0)})
        mappingf.update({"low": self.attrs.get("low" ,0.0)})
        mappingf.update({"statements": TemplateArrayDropoutLoop.replace("[statements]", TemplateStatements.format(**mapping))})
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
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)