import string
from inspect import cleandoc
from itertools import zip_longest
import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type



@onnx_op('Not')
class Not(BackendHandler):
    
    @classmethod
    def instantiate(cls, node, **kwargs):
        # check broadcast
        i = node.input_tensor[0]
        output_value = {node.valid_var_name(node.outputs[0]):
                        np.ones(shape=i.shape, dtype=i.dtype)}
        output_tensor = namedtupledict('output_tensor', output_value.keys())(**output_value)
        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, **kwargs)
    

    @classmethod
    def get_param_type_name(cls):
        return   'NotOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['not.c']


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ['stdbool.h']


    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            '''
            typedef struct {
                char* name;
                int ndim;
                int* shape;
                void *value;
            } NotOpParam;
            ''')

    
    def generate_c_code(self, **kwargs):
        res =''

        # include header
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        # 1
        TemplateArrayNotLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]))


        # 2
        TemplateStatements = '{Y}{StatementDims} = !({X}{StatementDims});'

        mapping = {}
        mapping.update({'X': self.input_tensor_names[0]})
        mapping.update({'Y': self.output_tensor_names[0]})


        StatementDims = ''
        for _, step in zip_longest(self.input_tensor[0].shape[::-1],  
                                   reversed(string.ascii_lowercase[8:8 + self.output_tensor_ndims[0]])):
            StatementDims  =  '[{0}]'.format(step) + StatementDims
        mapping.update({'StatementDims': StatementDims})        
        
        
        out_c_type = data_type.np2c(self.output_tensor_dtypes[0])

        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param,{t} {X}{Dims} , {t} {Y}{Dims}, void *inputs_params, void* outputs_params)
        {{
        {statements}
        }}
        ''')
        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'Y': self.output_tensor_names[0]})
        mappingf.update({'Dims':c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'statements': TemplateArrayNotLoop.replace('[statements]', TemplateStatements.format(**mapping))})
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
        mapping .update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping .update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc('''
        {indent}// define input & output
        {indent}{node_param_name}.ndim = {ndim};
        {indent}{node_param_name}.shape= OpShapeNode{node_num};
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

