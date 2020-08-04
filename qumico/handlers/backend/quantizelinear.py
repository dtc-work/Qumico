import string
from inspect import cleandoc
from itertools import zip_longest
import numpy as np

from onnx.backend.base import namedtupledict

from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type
from .math_mixin import ArithmeticMixin


@onnx_op('QuantizeLinear')
class QuantizeLinear(ArithmeticMixin, BackendHandler):

    OpenMP = False
    Quantizable = True

    @classmethod
    def instantiate(cls, node, **kwargs):
        x, y_scale, y_zero_point = node.input_tensor_values
        pre_sat = np.around(x / y_scale) + y_zero_point

        if np.issubdtype(y_zero_point.dtype, np.unsignedinteger):
            y = np.clip(pre_sat, 0, 255).astype(dtype=np.uint8)

        else:
            y = np.clip(pre_sat, -128, 127).astype(dtype=np.int8)

        output_value = {node.valid_var_name(node.outputs[0]):
                        np.ones(shape=y.shape, dtype=y.dtype)}
        output_tensor = namedtupledict('output_tensor', output_value.keys())(**output_value)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and
                QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, **kwargs)

    @classmethod
    def get_param_type_name(cls):
        return 'QuantizeLinearOpParam'

    @classmethod
    def get_c_op_file_name(cls):
        return ['quantizelinear.c']

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
            } QuantizeLinearOpParam;
            ''')

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res =cleandoc('''
        #define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
        #define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
        ''')
        return res
        

    def generate_c_code(self, **kwargs):
        res = ''
        # include header
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res += '\n\n'

        # param type
        res += self.get_c_param_type()
        res += '\n\n'
        res += self.get_op_variale_def()
        res += '\n\n'
        # 1
        TemplateArrayAddLoop = c_helper.generate_ndim_for_loop(np.empty(self.output_tensor_shapes[0]), pragma=self.OpenMP)
        if self.OpenMP:
            TemplateArrayAddLoop=TemplateArrayAddLoop.replace('[pragma]', self.PRAGMA_OMP)

        # 2
        mapping = {}
        # TemplateStatements = 'temp_arr{CStatementDims} = ROUND({X}{XStatementDims} / {Y}{YStatementDims}) + {Z}{ZStatementDims};\n'
        #
        # if data_type.np2c(self.output_tensor_dtypes[0]) == 'uint8_t':
        #     TemplateStatements += '            {C}{CStatementDims} = CLAMP(temp_arr{CStatementDims}, 0, 255);\n'
        # else:
        #     TemplateStatements += '            {C}{CStatementDims} = CLAMP(temp_arr{CStatementDims}, -127, 128);\n'

        if data_type.np2c(self.output_tensor_dtypes[0]) == 'uint8_t':
            TemplateStatements = '{C}{CStatementDims} = CLAMP(ROUND({X}{XStatementDims} / {Y}{YStatementDims}) + {Z}{ZStatementDims}, 0, 255);\n'
        else:
            TemplateStatements = '{C}{CStatementDims} = CLAMP(ROUND({X}{XStatementDims} / {Y}{YStatementDims}) + {Z}{ZStatementDims}, -127, 128);\n'

        mapping.update({'X': self.input_tensor_names[0]})
        mapping.update({'Y': self.input_tensor_names[1]})
        mapping.update({'Z': self.input_tensor_names[2]})
        mapping.update({'C': self.output_tensor_names[0]})

        XStatementDims = ''
        YStatementDims = ''
        ZStatementDims = ''
        CStatementDims = ''

        X, Y, Z = self.input_tensor_values

        for element_num_x, element_num_y, element_num_z, step in zip_longest(X.shape[::-1],
                                                              Y.shape[::-1],
                                                              Z.shape[::-1],
                                                              reversed(string.ascii_lowercase[
                                                                       8:8 + self.output_tensor_ndims[0]])):
            if element_num_x is not None:
                if element_num_x == 1:
                    XStatementDims = '[0]' + XStatementDims
                else:
                    XStatementDims = '[{0}]'.format(step) + XStatementDims

            if element_num_y is not None:
                if element_num_y == 1:
                    YStatementDims = '[0]' + YStatementDims
                else:
                    YStatementDims = '[{0}]'.format(step) + YStatementDims

            if element_num_z is not None:
                if element_num_z == 1:
                    ZStatementDims = '[0]' + ZStatementDims
                else:
                    ZStatementDims = '[{0}]'.format(step) + ZStatementDims

            CStatementDims = '[{0}]'.format(step) + CStatementDims

        mapping.update({'XStatementDims': XStatementDims})
        mapping.update({'YStatementDims': YStatementDims})
        mapping.update({'ZStatementDims': ZStatementDims})
        mapping.update({'CStatementDims': CStatementDims})

        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param,{x_type} {X}{XDims} , {y_type} {Y}{YDims}, {z_type} {Z}{ZDims}, {c_type} {C}{CDims}, void *inputs_params, void* outputs_params)
        {{      
        {statements}
        }}
        ''')
        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'Y': self.input_tensor_names[1]})
        mappingf.update({'Z': self.input_tensor_names[2]})
        mappingf.update({'C': self.output_tensor_names[0]})
        mappingf.update({'XDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({'YDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])})
        mappingf.update({'ZDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[2])})
        mappingf.update({'CDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({'x_type': data_type.np2c(self.node.input_tensor_values[0].dtype)})
        mappingf.update({'y_type': data_type.np2c(self.node.input_tensor_values[1].dtype)})
        mappingf.update({'z_type': data_type.np2c(self.node.input_tensor_values[2].dtype)})
        mappingf.update({'c_type': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update(
            {'statements': TemplateArrayAddLoop.replace('[statements]', TemplateStatements.format(**mapping))})
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
        mapping.update({'shape': ','.join(map(str, shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)

    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc = cleandoc('''
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
        mapping.update({'ndim': str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent': ' ' * indent})

        return TemplateInitFunc.format(**mapping)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
