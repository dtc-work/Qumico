from inspect import cleandoc
import numpy as np

from onnx import numpy_helper
from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type


@onnx_op("Identity")
class Identity(BackendHandler):
    
    @classmethod
    def instantiate(cls, node, **kwargs):
        
        i = node.input_tensor_values[0]
        output_value = {node.valid_var_name(node.outputs[0]):
                        np.ones(shape=i.shape, dtype=i.dtype)}
        output_tensor =namedtupledict("output_tensor", output_value.keys())(**output_value)
        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, **kwargs)
    

    @classmethod
    def get_param_type_name(cls):
        return   "IdentityOpParam"


    @classmethod
    def get_c_op_file_name(cls):
        return ["identity.c"]


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
            } IdentityOpParam;
            """)

    
    def generate_c_code(self, **kwargs):
        TEMPALTE_IDENTITY_FUNC = cleandoc("""
        void {op_func_name}(void *op_param, {t} Input{InputDims}, {t} Output{OutputDims}, void *inputs_params, void* outputs_params){{        
            memcpy(Output, Input, sizeof({t}) * {cumdim});
        }}
        """)

        res = ""
        res += self.get_c_param_type() # call only once
        res += "\n\n\n"

        # constant function
        mapping ={}
        mapping.update({"op_func_name": self.get_func_name()})
        mapping.update({"t": data_type.np2c(self.input_tensor_dtypes[0])})
        mapping.update({"cumdim": np.cumproduct(self.input_tensor_shapes[0])[-1]})
        mapping.update({"Input": self.input_tensor_names[0]})
        mapping.update({"Output": self.output_tensor_names[0]})        
        mapping.update({"InputDims": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mapping.update({"OutputDims": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})

        res += TEMPALTE_IDENTITY_FUNC.format(**mapping)
        
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
        {indent}{node_param_name}.value =&{node_param_name};
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
         {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        """)

        mapping = {}
        mapping.update({"node_param_name": node.node_param_name})
        mapping.update({"node_num": str(node_num)})
        mapping.update({"node_param_name": node.node_param_name})
        mapping.update({"ndim":str(self.output_tensor_ndims[0])})
        mapping.update({"output_val_name": self.output_tensor_names[0]})
        mapping.update({"indent":" " * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)