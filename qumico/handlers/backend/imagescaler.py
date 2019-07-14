from inspect import cleandoc
from collections import OrderedDict

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op("ImageScaler")

class ImageScaler(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]

        attrs = node.attrs
        attrs.update({"scale": attrs.get("scale", 1.0)})
        outputs_shape = input_data1.shape
        outputs_dtype = input_data1.dtype
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict("output_tensor", outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return "ImageScalerOpParam"


    @classmethod
    def get_c_op_file_name(cls):
        return ["imagescaler.c"]


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
            } ImageScalerOpParam;
            """)


    def generate_c_code(self, **kwargs):
        res =""
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"

        TemplateStatements = """
            int Y_n = {d1};
            int Y_c = {d2};
            int Y_h = {d3};
            int Y_w = {d4};

            const double bias[] = {bias};
            const double scale = {scale};

            int n;
            int c, h, w;

            for (n=0; n<Y_n; n++) {{
                for (c=0; c<Y_c; c++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            Y[n][c][h][w] = scale * X[n][c][h][w] + bias[c];
                        }}
                    }}
                }}
            }}
        """

        mapping = {}
        mapping.update({"d1": self.input_tensor_shapes[0][0]})
        mapping.update({"d2": self.input_tensor_shapes[0][1]})
        mapping.update({"d3": self.input_tensor_shapes[0][2]})
        mapping.update({"d4": self.input_tensor_shapes[0][3]})
        mapping.update({"bias": str(self.attrs["bias"]).replace('[', '{').replace(']', '}')})
        mapping.update({"scale": self.attrs["scale"]})

        # 3        
#        void {op_func_name}(void *op_param, {t} X{dims_X}, {t} bias{dims_bias}, {t} scale, {t} Y{dims}, void *inputs_params, void* outputs_params) {{
        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param, {t} X{dims_X}, {t} Y{dims}, void *inputs_params, void* outputs_params) {{
            {statements}
        }}
        """)

        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})
        mappingf.update({"dims_X": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
#        mappingf.update({"dims_bias": c_helper.generate_dim_bracket(np.array(self.attrs['bias']).shape)}) 
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
