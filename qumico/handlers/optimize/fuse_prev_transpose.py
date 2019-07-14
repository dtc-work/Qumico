from inspect import cleandoc
from collections import OrderedDict
import copy 

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.common import c_helper, data_type
from qumico.handlers.backend.transpose import Transpose
from qumico.handlers.optimize_handler import OptimizeHandler
from qumico.handlers.backend_handler import BackendHandler



class FusePrevTranspose(OptimizeHandler):

    OpenMP=False

    def __init__(self, prev_node, post_node):
        super(FusePrevTranspose, self).__init__()
        self.prev_node = copy.copy(prev_node)
        self.post_node = copy.copy(post_node)
        self.OpenMP = self.prev_node.op.OpenMP # should get from device info

        # remake input_tensor
        field_values = OrderedDict()
        field_values.update({self.prev_node.op.input_tensor_names[0]: self.prev_node.op.input_tensor_values[0]})
        for k,v in self.post_node.op.input_tensor_dict.items():
            if not k == self.prev_node.op.output_tensor_names[0]:
                field_values.update({k:v})
        self._input_tensor = namedtupledict("input_tensor", field_values.keys())(**field_values)
        
        self._output_tensor = self.post_node.op.output_tensor


    @classmethod
    def validate(cls, prev_node, post_node):
        return (issubclass(prev_node.op.__class__, BackendHandler) and
                prev_node.op.__class__ ==Transpose and
                issubclass(post_node.op.__class__, FusePrevTranspose))

  
    @classmethod
    def get_param_type_name(cls):
        return "FusePrevTransposeOpParam"

    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return []

    @classmethod
    def get_c_op_file_name(cls):
        return ["fuseprevtranspose.c"]


    @classmethod
    @OptimizeHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            """
            typedef struct {
                char* name;
                int ndim;
                int* shape;
                void *value;
            } FusePrevTransposeOpParam;
            """)



    def generate_c_code(self, **kwargs):
        res =""
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"

        # 1
        TemplateArrayFuseLoop = c_helper.generate_ndim_for_loop(np.empty(self.output_tensor_shapes[0]),
                                                                     pragma=self.OpenMP)
        if self.OpenMP:
            TemplateArrayFuseLoop=TemplateArrayFuseLoop.replace("[pragma]", self.PRAGMA_OMP)

        # param type
        res += self.get_c_param_type()
        res +="\n\n"
        
        premap = self.prev_node.op.generate_kernel_map() # transpose
        postmap = self.post_node.op.generate_kernel_map()
        postmap.update({"X": premap["X"]})
        postmap.update({"XStatementDims": premap["XStatementDims"]})
        template = self.post_node.op.generate_kernel_template()        
        statements =template.format(**postmap)

        # 3
        post_input_count = len(self.post_node.input_tensor)
        if post_input_count == 1:
            TemplateFunction = cleandoc("""
            void {op_func_name}(void *op_param, {t} {X}{dims_i}, {t} {C}{dims_o}, void *inputs_params, void* outputs_params) {{        
                {statements}
            }}
            """)
    
            mappingf = {}
            mappingf.update({"op_func_name": self.get_func_name()})
            mappingf.update({"dims_i": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
            mappingf.update({"dims_o": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
            mappingf.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
            mappingf.update({"X": self.input_tensor_names[0]})
            mappingf.update({"C": self.output_tensor_names[0]})
            mappingf.update({"statements": TemplateArrayFuseLoop.replace("[statements]", statements)})
            res += "\n\n"
            res += TemplateFunction.format(**mappingf)

        elif post_input_count ==2:
            TemplateFunction = cleandoc("""
            void {op_func_name}(void *op_param, {t} {X1}{dims_i1}, {t} {X2}{dims_i2}, {t} {C}{dims_o}, void *inputs_params, void* outputs_params) {{        
                {statements}
            }}
            """)

            mappingf = {}
            mappingf.update({"op_func_name": self.get_func_name()})
            mappingf.update({"dims_i1": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
            mappingf.update({"dims_i2": c_helper.generate_dim_bracket(self.input_tensor_shapes[1])})
            mappingf.update({"dims_o": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
            mappingf.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
            mappingf.update({"X1": self.input_tensor_names[0]})
            mappingf.update({"X2": self.input_tensor_names[1]})
            mappingf.update({"C": self.output_tensor_names[0]})
            mappingf.update({"statements": TemplateArrayFuseLoop.replace("[statements]", statements)})
            res += "\n\n"
            res += TemplateFunction.format(**mappingf)
        else:
            raise ValueError()

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
        mapping.update({"add_name": self.name})
        mapping.update({"ndim":str(self.output_tensor_ndims[0])})
        mapping.update({"output_val_name": self.output_tensor_names[0]})
        mapping.update({"indent":" " * indent})

        return TemplateInitFunc.format(**mapping)
    