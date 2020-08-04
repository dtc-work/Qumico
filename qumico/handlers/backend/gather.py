import string
import textwrap
from inspect import cleandoc

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

""" design

input [A][B][C][D][E]
index [F][G][H]
case: axis=0
    input: right:None
           replace: A
           left:B,C,D,E
    output[F][G][H][B][C][D][E] = input[index[F][G][H]][B][C][D][E]

case: axis=1
    input: right:A
           replace: B
           left:C,D,E
    output: right + index + left
         output[A][F][G][H][C][D][E] = input[A][index[F][G][H]][C][D][E]

case: axis=2
    input: right:A, B
           replace: C
           left:D,E 
    output: right + index + left
         output[A][B][F][G][H][D][E] = input[A][B][index[F][G][H]][D][E]
"""

@onnx_op('Gather')
class Gather(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):

        input_data1 = node.input_tensor[0]
        input_data2 = node.input_tensor[1]
        # parms
        if (node.attrs.get('axis') is None):
            node.attrs['axis'] = 0

        output_shape_tmp = list(input_data1.shape)
        for index, s in enumerate(input_data2.shape):
            if index==0:
                output_shape_tmp[node.attrs['axis']] = s
            else:
                output_shape_tmp.insert(node.attrs['axis']+index, s)

        output_shape = tuple(output_shape_tmp)
        output_dtype = input_data1.dtype
        output_dict ={node.valid_var_name(node.outputs[0]): np.empty(shape=output_shape, dtype=output_dtype)}
        output_tensor = namedtupledict('output_tensor', output_dict.keys())(**output_dict)

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, attrs=node.attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return   'GatherOpParam'    

    @classmethod
    def get_c_op_file_name(cls):
        return ['gather.c']

    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return []

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            '''
            typedef struct {
                char* name;
                int   axis;
            } GatherOpParam;
            ''')

    def generate_c_code(self, **kwargs):
        axis = self.attrs['axis']
        data_ndims = self.input_tensor_ndims[0]
        output_ndims = self.output_tensor_ndims[0]
        indices_ndims = self.input_tensor_ndims[1]

        res =''
        
        # include header
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        textwrap.TextWrapper()
        TemplateKernel =textwrap.indent( 
        """
        if ({array_indices}>= 0) {{
            {array_output}= {array_data};
        }} else {{
            {array_output} = 0.0;
        }}
        """,prefix=" " * 4 * output_ndims)

        TemplateLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]))
                
        array_right_data = ''.join(['[' + v + ']' for v in string.ascii_lowercase[8:8+axis]])
        array_left_data = ''.join(['[' + v + ']' for v in string.ascii_lowercase[9+axis+indices_ndims-1:8+data_ndims+indices_ndims-1]])

        array_indices ="indices"  + "".join(['[' + v + ']' for v in string.ascii_lowercase[8+axis:8+axis+indices_ndims]])
        array_data = "data" + array_right_data + '[' + array_indices + ']' + array_left_data
        array_output ="output" + ''.join(['[' + v + ']' for v in string.ascii_lowercase[8:8+output_ndims]])

        mapping_kernel = {}
        mapping_kernel.update({"array_data":array_data})
        mapping_kernel.update({"array_indices":array_indices})
        mapping_kernel.update({"array_output":array_output})
 
        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t1} data{dims_data}, {t2} indices{dims_indices}, {t1} output{dims_output}, void *inputs_params, void* outputs_params) {{
            {loop_statements}
        }}

        ''')
 
        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'dims_data': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({'dims_indices': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])})
        mappingf.update({'dims_output': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({'t1': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'t2': data_type.np2c(self.input_tensor_dtypes[1])})
        mappingf.update({'loop_statements': TemplateLoop.replace('[statements]', TemplateKernel.format(**mapping_kernel))})
        res += '\n\n'
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        TemplateVariavbles = cleandoc('''
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            ''')

        ndim = self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]

        mapping = {}
        mapping.update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping) 
    
    


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc('''
        {indent}// define input & output
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
        {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        ''')

        mapping = {}
        mapping.update({'node_param_name': node.node_param_name})
        mapping.update({'node_num': str(node_num)})
        mapping.update({'add_name': self.get_name()})
        mapping.update({'ndim':str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent':' ' * indent})

        return TemplateInitFunc.format(**mapping)    
  
    
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
