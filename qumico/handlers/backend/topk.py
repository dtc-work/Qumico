from inspect import cleandoc
import logging
import math
import string

import numpy as np

from onnx.backend.base import namedtupledict
from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op


@onnx_op('TopK')
class TopK(BackendHandler):

    OpenMP=False


    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]
        attrs = node.attrs

        if (attrs.get('axis') is None):       # default is -1.
            attrs['axis'] = -1
        if (attrs['axis'] < 0):
            attrs['axis'] = node.input_tensor_ndims[0] + attrs['axis']     # translate 'axis' to positive value
        if (attrs.get('largest') is None):       # default is 1.
            attrs['largest'] = 1
        if (attrs.get('sorted') is None):       # default is 1.
            attrs['sorted'] = 1

        outputs_dtype = input_data1.dtype
        outputs_shape_tmp = list(node.input_tensor_shapes[0])
        outputs_shape_tmp[attrs['axis']] = node.input_tensor_values[1][0]
        outputs_shape = tuple(outputs_shape_tmp)

        try:
            outputs_tensor_tmp = np.ones(shape=outputs_shape, dtype=outputs_dtype)
        except Exception as e:
            logging.warn('use model output shape in TopK op because of shape error:{0}'.format(e))
            outputs_shape = node.outputs_info[0][1]

        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype),
                        node.valid_var_name(node.outputs[1]): np.ones(shape=outputs_shape, dtype=np.int)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and 
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True
        
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return 'TopKOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['topk.c']


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ["stdlib.h"]
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
                int   axis;
                int   largest;
                int   sorted;
            }} TopKOpParam;
            '''
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res = '// get_op_variale_def'
        return res

    def generate_c_code(self, **kwargs):
        axis = self.attrs['axis']
        value_ndims = self.output_tensor_ndims[0]
        value_shapes = self.output_tensor_shapes[0]

        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        TemplateStatements = '''
        const int  axis = {axis};
        const int  largest = {largest};
        const int  sorted = {sorted};
    
        const int  X_shape[] = {X_shape};
        const int  K_val = K[0];
        const int  Values_shape[] = {Values_shape};
        const int  Indices_shape[] = {Indices_shape};

        int        sorted_indices{dims_X};

        '''
        mapping = {}
        mapping.update({'op_func_name': self.get_func_name()})
        mapping.update({'axis': self.attrs['axis']})
        mapping.update({'largest': self.attrs['largest']})
        mapping.update({'sorted': self.attrs['sorted']})
        mapping.update({'dims_X': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mapping.update({'X_shape': str(self.input_tensor_shapes[0]).replace('(','{').replace(')','}')})
        mapping.update({'Values_shape': str(self.output_tensor_shapes[0]).replace('(','{').replace(')','}')})
        mapping.update({'Indices_shape': str(self.output_tensor_shapes[1]).replace('(','{').replace(')','}')})
        mapping.update({'target_range': self.input_tensor_shapes[0][self.attrs['axis']]})
        mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})

        TemplatePreProcessLoop = c_helper.generate_ndim_for_loop(np.ones(self.input_tensor_shapes[0]))

        TemplatePreProcessCore = '''
            sorted_indices{dims_all} = {target_rank};
        '''


        TemplateArrayLoopLeft = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0][0:axis+1])) if (axis >= 0) else '[statements]'
        TemplateArrayLoopRight = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0][axis+1:]),gen=iter(string.ascii_lowercase[9+axis:])) if (axis < value_ndims-1) else '[statements]'

        TemplateLoopCore = '''
                for (int z={target_rank}+1; z<{target_range}; z++) {{
        '''
        if (self.attrs['largest'] == 1):
            TemplateLoopCore += '''
                        if (X{dims_left}[sorted_indices{dims_left}[{target_rank}]{dims_right}]{dims_right} < X{dims_left}[sorted_indices{dims_left}[z]{dims_right}]{dims_right}) {{
            '''
        else:
            TemplateLoopCore += '''
                        if (X{dims_left}[sorted_indices{dims_left}[{target_rank}]{dims_right}]{dims_right} > X{dims_left}[sorted_indices{dims_left}[z]{dims_right}]{dims_right}) {{
            '''
        TemplateLoopCore += '''
                        int tmp_idx = sorted_indices{dims_all};
                        sorted_indices{dims_all} = sorted_indices{dims_left}[z]{dims_right};
                        sorted_indices{dims_left}[z]{dims_right} = tmp_idx;
                    }}
                }}
        '''

        mapping_loop = {}
        mapping_loop.update({'dims_all': ''.join(['[' + v + ']'  for v in  string.ascii_lowercase[8:8+value_ndims]])}) 
        mapping_loop.update({'dims_left': ''.join(['[' + v + ']'  for v in  string.ascii_lowercase[8:8+axis]])}) 
        mapping_loop.update({'dims_right': ''.join(['[' + v + ']'  for v in  string.ascii_lowercase[9+axis:8+value_ndims]])}) 
        mapping_loop.update({'target_range': self.input_tensor_shapes[0][axis]})
        mapping_loop.update({'target_rank': string.ascii_lowercase[8+axis]})

        TemplatePostProcessLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]))

        TemplatePostProcessCore = '''
                Values{dims_all} = X{dims_left}[sorted_indices{dims_left}[{target_rank}]{dims_right}]{dims_right};
                Indices{dims_all} = sorted_indices{dims_left}[{target_rank}]{dims_right};
        '''

        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} X{dims_X}, long long int K[], {t} Values{dims_Values}, long long int Indices{dims_Indices}, void *inputs_params, void* outputs_params) {{
            {pre_statements}
            {preloop_statements}
            {loop_statements}
            {postloop_statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'dims_X': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({'dims_Values': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({'dims_Indices': c_helper.generate_dim_bracket(self.output_tensor_shapes[1])})
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'pre_statements': TemplateStatements.format(**mapping)})
        mappingf.update({'preloop_statements': TemplatePreProcessLoop.replace('[statements]', TemplatePreProcessCore.format(**mapping_loop))})
        mappingf.update({'loop_statements': TemplateArrayLoopLeft.replace('[statements]', TemplateArrayLoopRight.replace('[statements]', TemplateLoopCore.format(**mapping_loop)))})
        mappingf.update({'postloop_statements': TemplatePostProcessLoop.replace('[statements]', TemplatePostProcessCore.format(**mapping_loop))})
        res += '\n\n'
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        return ""

    def gen_init_func(self, node, node_num, indent=4, **kwargs):
        return ""

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
