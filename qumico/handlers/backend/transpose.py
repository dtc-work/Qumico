import string
from inspect import cleandoc
from itertools import zip_longest

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import data_type
from qumico.common import c_helper


@onnx_op('Transpose')
class Transpose(BackendHandler):

    OpenMP=False
       
    @classmethod
    def instantiate(cls, node, **kwargs):
        i = node.input_tensor_values[0]
        
        transposed = np.transpose(i, node.attrs.get('perm'))
        output_value = {node.valid_var_name(node.outputs[0]): 
                        np.ones(shape=transposed.shape, dtype=i.dtype)}
        output_tensor =namedtupledict('output_tensor', output_value.keys())(**output_value)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and 
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor,
                   attrs=node.attrs, **kwargs)

    @classmethod
    def get_param_type_name(cls):
        return   'TransposeOpParam'

    @classmethod
    def get_c_op_file_name(cls):
        return ['transpose.c']

    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ['stdio.h']

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
            } TransposeOpParam;
            ''')


    def generate_kernel_map(self):
        mapping = {}
        mapping.update({"X": self.input_tensor_names[0]})
        mapping.update({"C": self.output_tensor_names[0]})

        XStatementDims = ""
        CStatementDims = ""

        X = self.input_tensor_values[0]
        Y = self.output_tensor_values[0]
        output_ndim = self.output_tensor_ndims[0]
        input_ndim = self.input_tensor_ndims[0]
        transpose_perm = self.attrs.get("perm") or list(range(output_ndim))[::-1]
        step_vars = string.ascii_lowercase[8:8 + self.output_tensor_ndims[0]]

        for i, (element_num_y, step) in enumerate(zip_longest(Y.shape[::-1], reversed(step_vars))):
            
            element_num_x = X.shape[-i-1]            
            if element_num_y is not None :
                if element_num_y == 1:
                    CStatementDims  = "[0]" + CStatementDims 
                else:
                    CStatementDims  =  "[{0}]".format(step) + CStatementDims
                if  element_num_x == 1:
                    XStatementDims  = "[0]" + XStatementDims 
                else:
                    var_index = transpose_perm.index(input_ndim-i-1)
                    XStatementDims =  "[{0}]".format(step_vars[var_index]) +  XStatementDims 

        mapping.update({"XStatementDims": XStatementDims})
        mapping.update({"CStatementDims": CStatementDims})
        return mapping


    def generate_kernel_code(self):
        template = "{C}{CStatementDims} = {X}{XStatementDims};"
        return template.format(**self.generate_kernel_map())


    def generate_c_code(self, **kwargs):
        res =''

        # include header
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'        

        # 1
        TemplateArrayTransposeLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]),
                                                                     pragma=self.OpenMP)
        
        if self.OpenMP:
            TemplateArrayTransposeLoop=TemplateArrayTransposeLoop.replace('[pragma]', self.PRAGMA_OMP)


        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} {X}{XDims}, {t} {C}{CDims}, void *inputs_params, void* outputs_params)
        {{
        {statements}
        }}
        ''')
        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'C': self.output_tensor_names[0]})
        mappingf.update({'XDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({'CDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'statements': TemplateArrayTransposeLoop.replace('[statements]', self.generate_kernel_code())})
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
        mapping.update({'add_name': self.name})
        mapping.update({'ndim':str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent':' ' * indent})

        return TemplateInitFunc.format(**mapping)
    
    @classmethod
    def version_1(cls, node, **kwargs):
            return cls.instantiate(node, **kwargs)