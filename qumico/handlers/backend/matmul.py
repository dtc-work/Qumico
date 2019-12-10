import string
from inspect import cleandoc
from collections import OrderedDict

import numpy as np

from onnx.backend.base import namedtupledict
from onnx import numpy_helper

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op('MatMul')

class MatMul(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]
        input_data2 = node.input_tensor[1]
        max_dim = 5
        if (input_data1.ndim > max_dim or input_data2.ndim > max_dim):
            raise ValueError()
        if (input_data1.ndim >= 2 and input_data2.ndim >= 2):
            if (input_data1.shape[-1] != input_data2.shape[-2]):
                raise ValueError()
#        input_data1_new_shape = (1,) * (max_dim-input_data1.ndim) + input_data1.shape
#        if (input_data2.ndim == 1):
#            input_data2_new_shape = ((1,) * (max_dim-2), input_data2.shape, 1)
#        else:
#            input_data2_new_shape = (1,) * (max_dim-input_data2.ndim) + input_data2.shape
#        input_data1 = input_data1.reshape(input_data1_new_shape)
#        input_data2 = input_data2.reshape(input_data2_new_shape)
        inputs_dict = OrderedDict()
        inputs_dict.update({'A': input_data1})
        inputs_dict.update({'B': input_data2})

        tmp_output_shape = []
        max_dim = max(input_data1.ndim, input_data2.ndim) 
        if (max_dim >= 2):
            for d in range(-max_dim, -2):
                if (-d > input_data1.ndim):
                    tmp_output_shape.append(input_data2.shape[d])
                elif (-d > input_data2.ndim):
                    tmp_output_shape.append(input_data1.shape[d])
                else:
                    tmp_output_shape.append(max(input_data1.shape[d], input_data2.shape[d]))
            if (input_data1.ndim != 1):
                tmp_output_shape.append(input_data1.shape[-2])
            if (input_data2.ndim != 1):
                tmp_output_shape.append(input_data2.shape[-1])
        
        outputs_shape = tuple(tmp_output_shape)
        outputs_dtype = input_data1.dtype  if input_data1.dtype  == input_data2.dtype else np.double
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return 'MatMulOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['matmul.c']


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
            }} MatMulOpParam;
            '''
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'
        
        input_shapes = []
        input_shapes.append(self.input_tensor_shapes[0])
        input_shapes.append(self.input_tensor_shapes[1])
        input_mod_shapes = []

        max_dim = 5
        if (len(input_shapes[0]) == 1):
            input_mod_shapes.append(((1,) * (max_dim-1) + input_shapes[0]))
        else:
            input_mod_shapes.append((1,) * (max_dim-len(input_shapes[0])) + input_shapes[0])
        if (len(input_shapes[1]) == 1):
            input_mod_shapes.append(((1,) * (max_dim-2) + input_shapes[1] + (1,)))
        else:
            input_mod_shapes.append((1,) * (max_dim-len(input_shapes[1])) + input_shapes[1])
        outputs_shape =(max(input_mod_shapes[0][0], input_mod_shapes[1][0]),
                        max(input_mod_shapes[0][1], input_mod_shapes[1][1]),
                        max(input_mod_shapes[0][2], input_mod_shapes[1][2]),
                        input_mod_shapes[0][3],
                        input_mod_shapes[1][4])

        output_names = self.output_tensor_names[0]

        ndim = self.output_tensor_ndims[0]

        TemplateStatements = '''
            int   A_h = {A_d0};
            int   A_i = {A_d1};
            int   A_j = {A_d2};
            int   A_m = {A_d3};
            int   A_k = {A_d4};
            int   B_h = {B_d0};
            int   B_i = {B_d1};
            int   B_j = {B_d2};
            int   B_k = {B_d3};
            int   B_n = {B_d4};
            int   Y_h = {Y_d0};
            int   Y_i = {Y_d1};
            int   Y_j = {Y_d2};
            int   Y_m = {Y_d3};
            int   Y_n = {Y_d4};

            {t} *_A = ({t} *)A;
            {t} *_B = ({t} *)B;
            {t} *_Y = ({t} *)Y;
            {t} tmpA, tmpB;

            int   h, i, j;  
            int   k;
            int   m;
            int   n;

            memset( Y, ({t})0.0, sizeof(*_Y)*Y_h*Y_i*Y_j*Y_m*Y_n );

            for (h=0; h < Y_h; h++) {{
                for (i=0; i < Y_i; i++) {{
                    for (j=0; j < Y_j; j++) {{
                        for (m=0; m < Y_m; m++) {{
                            for (n=0; n < Y_n; n++) {{
                                for (k=0; k < B_k; k++) {{
                                    tmpA = *(_A + h*(Y_i*Y_j*Y_m*B_k) + i*(Y_j*Y_m*B_k) + j*(Y_m*B_k) + m*(B_k) + k);
                                    tmpB = *(_B + h*(Y_i*Y_j*B_k*Y_n) + i*(Y_j*B_k*Y_n) + j*(B_k*Y_n) + k*(Y_n) + n);
                                    *(_Y + h*(Y_i*Y_j*Y_m*Y_n) + i*(Y_j*Y_m*Y_n) + j*(Y_m*Y_n) + m*(Y_n) + n) += tmpA * tmpB;
//                                    Y[h][i][j][m][n] += A[h][i][j][m][k] * B[h][i][j][k][n];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        '''

        mapping = {}
        mapping.update({'A_d0': input_mod_shapes[0][0]})
        mapping.update({'A_d1': input_mod_shapes[0][1]})
        mapping.update({'A_d2': input_mod_shapes[0][2]})
        mapping.update({'A_d3': input_mod_shapes[0][3]})
        mapping.update({'A_d4': input_mod_shapes[0][4]})
        mapping.update({'B_d0': input_mod_shapes[1][0]})
        mapping.update({'B_d1': input_mod_shapes[1][1]})
        mapping.update({'B_d2': input_mod_shapes[1][2]})
        mapping.update({'B_d3': input_mod_shapes[1][3]})
        mapping.update({'B_d4': input_mod_shapes[1][4]})
        mapping.update({'Y_d0': outputs_shape[0]})
        mapping.update({'Y_d1': outputs_shape[1]})
        mapping.update({'Y_d2': outputs_shape[2]})
        mapping.update({'Y_d3': outputs_shape[3]})
        mapping.update({'Y_d4': outputs_shape[4]})
        mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})


        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} A{dims_A}, {t} B{dims_B}, {t} Y{dims}, void *inputs_params, void* outputs_params)
        {{
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'A': self.input_tensor_names[0]})
        mappingf.update({'dims_A': c_helper.generate_dim_bracket(input_shapes[0])}) 
        mappingf.update({'B': self.input_tensor_names[1]})
        mappingf.update({'dims_B': c_helper.generate_dim_bracket(input_shapes[1])}) 
        mappingf.update({'Y': self.output_tensor_names[0]})
        mappingf.update({'dims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'statements': TemplateStatements.format(**mapping)})
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
        mapping = {}
        mapping.update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):
#         TemplateInitFunc=cleandoc('''
#         {indent}// define input & output
#         {indent}{node_param_name}.ndim = {ndim};
#         {indent}{node_param_name}.shape= OpShapeNode{node_num};
#         {indent}{node_param_name}.value =&{add_name};
#         {indent}Nodes[{node_num}].op_param = &{node_param_name};
#         {indent}Nodes[{node_num}].outputs = &OutputNode{node_num};
#         {indent}Nodes[{node_num}].output_ndim = {ndim};
#         {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
#         ''')

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
        mapping.update({'ndim':self.output_tensor_ndims[0]})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent':' ' * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def need_c_headers(cls):
        return ['string.h']

    
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


    @classmethod
    def version_9(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


