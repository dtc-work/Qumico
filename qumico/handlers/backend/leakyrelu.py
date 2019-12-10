import string
from inspect import cleandoc

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op

from qumico.handlers.optimize.fuse_prev_transpose import FusePrevTranspose


@onnx_op('LeakyRelu')
class LeakyRelu(BackendHandler,FusePrevTranspose):

    OpenMP=False

    @classmethod
    def instantiate(cls, node, **kwargs):
        attr_alpha = node.attrs.get('alpha', 0.01)
        node.attrs['alpha']=attr_alpha
        input_data = node.input_tensor[0]
        outputs_shape = input_data.shape
        outputs_dtype = input_data.dtype
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True        
        
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=node.attrs)



    @classmethod
    def get_param_type_name(cls):
        return 'LeakyReluOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['leakyrelu.c']


    @classmethod
    def get_c_op_include_header(cls):
        return []
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
                {t} alpha;
            }} LeakyReluOpParam;
            '''
        )
        mapping = {
            't': 'double'
        }

        return TEMPLATE_STRUCT.format(**mapping)
    
    def generate_kernel_map(self, alpha_variable_value=True):
        mapping = {}
        mapping.update({"alpha": self.attrs["alpha"] if alpha_variable_value else "alpha"})
        mapping.update({"XStatementDims": "".join(["[" + v + "]"  for v in  string.ascii_lowercase[8:8+self.output_tensor_ndims[0]]])})
        mapping.update({"YStatementDims": "".join(["[" + v + "]"  for v in  string.ascii_lowercase[8:8+self.output_tensor_ndims[0]]])})
        mapping.update({"X": self.input_tensor_names[0]})
        mapping.update({"Y": self.output_tensor_names[0]})
        return mapping
        
    def generate_kernel_template(self):
        return """
                if ({X}{XStatementDims} > 0) {{
                    {Y}{YStatementDims} = {X}{XStatementDims};
                }} else {{
                    {Y}{YStatementDims} = {alpha} * {X}{XStatementDims};
                }}
                """
        
    def generate_kernel_code(self, alpha_variable_value=True):
        template = self.generate_kernel_template()
        return template.format(**self.generate_kernel_map(alpha_variable_value))        


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        # 1
        TemplateArrayLeakyReluLoop = c_helper.generate_ndim_for_loop(np.ones(self.output_tensor_shapes[0]),
                                                                     pragma=self.OpenMP)
        if self.OpenMP:
            TemplateArrayLeakyReluLoop=TemplateArrayLeakyReluLoop.replace('[pragma]', self.PRAGMA_OMP)

        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} {X}{dims}, {t} {Y}{dims}, void *inputs_params, void* outputs_params) {{
            LeakyReluOpParam *param_ptr = (LeakyReluOpParam *)op_param;
            const {t} alpha = {alpha};
        
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'dims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'alpha': self.attrs['alpha']})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'Y': self.output_tensor_names[0]})
        mappingf.update({'statements': TemplateArrayLeakyReluLoop.replace('[statements]', self.generate_kernel_code(alpha_variable_value=False))})
        res += '\n\n'
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        TemplateVariavbles = cleandoc('''
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            ''')
        ndim =  self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]
        attr_alpha = node.attrs['alpha']
        mapping = {}
        mapping.update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc('''
        {indent}// define input & output
        {indent}{node_param_name}.alpha = {alpha};
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
        mapping.update({'alpha': node.attrs['alpha']})

        return TemplateInitFunc.format(**mapping)

  
    @classmethod
    def version_6(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
