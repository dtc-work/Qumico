import string
from inspect import cleandoc
from collections import OrderedDict

import numpy as np

from onnx.backend.base import namedtupledict
from onnx import numpy_helper

from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op('QLinearMatMul')

class QLinearMatMul(BackendHandler):

    Quantizable = True
    OpenMP = False

    @classmethod
    def instantiate(cls, node, **kwargs):
        if (len(node.input_tensor)!=8):
            raise ValueError()

        input_data1 = node.input_tensor[0]
        input_data1_scale = node.input_tensor[1]
        input_data1_zero_point = node.input_tensor[2]
        input_data2 = node.input_tensor[3]
        input_data2_scale = node.input_tensor[4]
        input_data2_zero_point = node.input_tensor[5]
        output_data_scale = node.input_tensor[6]
        output_data_zero_point = node.input_tensor[7]

        max_dim = 5
        if (input_data1.ndim > max_dim or input_data2.ndim > max_dim):
            raise ValueError()
        
        if (input_data1_scale.ndim > 1 or input_data1_zero_point.ndim > 1):
            raise ValueError()
        elif (input_data1_scale.shape[0] != input_data1_zero_point.shape[0]):
            raise ValueError()
        elif (input_data1_scale.shape[0] > 1):
            if (input_data1_scale.shape[0] != input_data1.shape[-2]):
                raise ValueError()
        
        if (input_data2_scale.ndim > 1 or input_data2_zero_point.ndim > 1):
            raise ValueError()
        elif (input_data2_scale.shape[0] != input_data2_zero_point.shape[0]):
            raise ValueError()
        elif (input_data2_scale.shape[0] > 1):
            if (input_data2_scale.shape[0] != input_data2.shape[-1]):
                raise ValueError()

        if (output_data_scale.ndim > 1 or output_data_zero_point.ndim > 1):
            raise ValueError()
        elif (output_data_scale.shape[0] != output_data_zero_point.shape[0]):
            raise ValueError()
        elif (output_data_scale.shape[0] > 1):
            if (output_data_scale.shape[0] != input_data1.shape[-2]):
                raise ValueError()
        
        matmul_output = np.matmul(input_data1, input_data2)
        outputs_shape = matmul_output.shape
        outputs_dtype = input_data1.dtype if input_data1.dtype == input_data2.dtype else np.double
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.empty(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True

        return cls(node, input_tensor=node.input_tensor, output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return 'QLinearMatMulOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['qlinearmatmul.c']


    @classmethod
    def get_c_op_include_header(cls):
        return ['math.h']
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
            }} QLinearMatMulOpParam;
            '''
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)


    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res = ''
        res += '#define qlinearmatmul_CLAMP(x, low, high) ((x) > (high) ? (high) : ((x) < (low) ? (low) : (x)))' + '\n\n'
        res += 'int qlinearmatmul_ROUND(float x) {' + '\n'
        res += '    if (fabsf(x - (int)x) != 0.5) return (x >= 0.0 ? (int)(x + 0.5) : (int)(x - 0.5));' + '\n'
        res += '    else return (x >= 0.0 ? (int)((x + 0.5) - (int)(x + 0.5) % 2) : (int)((x - 0.5) - (int)(x - 0.5) % 2)); ' + '\n'
        res += '}'

        return res


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        # variable define
        res += self.get_op_variale_def()
        res += '\n\n'
        
        input_shapes = []
        input_shapes.append(self.input_tensor_shapes[0]) # a
        input_shapes.append(self.input_tensor_shapes[3]) # b
        input_mod_shapes = []
        input_org_shapes = []

        max_dim = 5
        if (len(input_shapes[0]) == 1):
            input_mod_shapes.append(((1,) * (max_dim-1) + input_shapes[0]))
            input_org_shapes.append(((0,) * (max_dim-1) + input_shapes[0]))
        else:
            input_mod_shapes.append((1,) * (max_dim-len(input_shapes[0])) + input_shapes[0])
            input_org_shapes.append((0,) * (max_dim-len(input_shapes[0])) + input_shapes[0])
        if (len(input_shapes[1]) == 1):
            input_mod_shapes.append(((1,) * (max_dim-2) + input_shapes[1] + (1,)))
            input_org_shapes.append(((0,) * (max_dim-2) + input_shapes[1] + (1,)))
        else:
            input_mod_shapes.append((1,) * (max_dim-len(input_shapes[1])) + input_shapes[1])
            input_org_shapes.append((0,) * (max_dim-len(input_shapes[1])) + input_shapes[1])

        outputs_shape = ((1,) * (max_dim-len(self.output_tensor_shapes[0]))) + self.output_tensor_shapes[0]

        output_names = self.output_tensor_names[0]
        ndim = self.output_tensor_ndims[0]
        zero_point_shapes = [self.input_tensor_shapes[2][0],self.input_tensor_shapes[5][0],self.input_tensor_shapes[7][0]]

        TemplateStatements = '''
            const int   A_h = {A_d0};
            const int   A_i = {A_d1};
            const int   A_j = {A_d2};
            const int   A_m = {A_d3};
            const int   A_k = {A_d4};
            const int   B_h = {B_d0};
            const int   B_i = {B_d1};
            const int   B_j = {B_d2};
            const int   B_k = {B_d3};
            const int   B_n = {B_d4};
            const int   Y_h = {Y_d0};
            const int   Y_i = {Y_d1};
            const int   Y_j = {Y_d2};
            const int   Y_m = {Y_d3};
            const int   Y_n = {Y_d4};

            const int   A_h_o = {A_d0_o};
            const int   A_i_o = {A_d1_o};
            const int   A_j_o = {A_d2_o};
            const int   B_h_o = {B_d0_o};
            const int   B_i_o = {B_d1_o};
            const int   B_j_o = {B_d2_o};

            {ta} *_A = ({ta} *)A;
            {tb} *_B = ({tb} *)B;
            {ty} *_Y = ({ty} *)Y;
            int tmpA, tmpB, tmpY;
            {tb} BT [{B_d0}][{B_d1}][{B_d2}][{B_d3}][{B_d4}];
            {tb} *_BT = ({tb} *)BT;

            {taz} a_zero_point_mod[{A_d3}];
            {tbz} b_zero_point_mod[{B_d4}];
            {tyz} y_zero_point_mod[{Y_d3}];

            float a_scale_mod[{A_d3}];
            float b_scale_mod[{B_d4}];
            float y_scale_mod[{Y_d3}];
            float multiplier;

            int   h, i, j;
            int   ah, ai, aj;
            int   bh, bi, bj;
            int   k;
            int   m;
            int   n;

            int   tmpA_pos_h, tmpA_pos_i, tmpA_pos;
            int   tmpB_pos_h, tmpB_pos_i, tmpB_pos;
            int   tmpY_pos_h, tmpY_pos_i, tmpY_pos;

            memset( Y, ({ty})0, sizeof(*_Y)*Y_h*Y_i*Y_j*Y_m*Y_n );
        '''

        if (zero_point_shapes[0]==1):
            TemplateStatements += '''
#pragma omp parallel for
            for (m=0; m < A_m; m++) {{
                a_zero_point_mod[m] = a_zero_point[0];
                a_scale_mod[m] = a_scale[0];
            }}
            '''
        else :
            TemplateStatements += '''
#pragma omp parallel for
            for (m=0; m < A_m; m++) {{
                a_zero_point_mod[m] = a_zero_point[m];
                a_scale_mod[m] = a_scale[m];
            }}
            '''
        if (zero_point_shapes[1]==1):
            TemplateStatements += '''
#pragma omp parallel for
            for (n=0; n < B_n; n++) {{
                b_zero_point_mod[n] = b_zero_point[0];
                b_scale_mod[n] = b_scale[0];
            }}
            '''
        else :
            TemplateStatements += '''
#pragma omp parallel for
            for (n=0; n < B_n; n++) {{
                b_zero_point_mod[n] = b_zero_point[n];
                b_scale_mod[n] = b_scale[n];
            }}
            '''
        
        if (zero_point_shapes[2]==1):
            TemplateStatements += '''
#pragma omp parallel for
            for (m=0; m < A_m; m++) {{
                y_zero_point_mod[m] = y_zero_point[0];
                y_scale_mod[m] = y_scale[0];
            }}
            '''
        else :
            TemplateStatements += '''
#pragma omp parallel for
            for (m=0; m < A_m; m++) {{
                y_zero_point_mod[m] = y_zero_point[m];
                y_scale_mod[m] = y_scale[m];
            }}
            '''    
            
        TemplateStatements += '''
            for (h=0; h < B_h; h++) {{
                bh = (B_h_o > 1) ? h : 0;
                tmpB_pos_h = bh*(B_i*B_j*B_k*B_n);
                for (i=0; i < B_i; i++) {{
                    bi = (B_i_o > 1) ? i : 0;
                    tmpB_pos_i = tmpB_pos_h + bi*(B_j*B_k*B_n);
                    for (j=0; j < B_j; j++) {{
                        bj =  (B_j_o > 1) ? j : 0;
                        tmpB_pos = tmpB_pos_i + bj*(B_k*B_n);
#pragma omp parallel for private(n,k)
                        for (n=0; n < B_n; n++) {{
                            for (k=0; k < B_k; k++) {{
                                *(_BT + tmpB_pos + n*(B_k) + k) = *(_B + tmpB_pos + k*(B_n) + n);
                            }}
                        }}

                    }}
                }}
            }}

            for (h=0; h < Y_h; h++) {{
                ah = (A_h_o > 1) ? h : 0;
                bh = (B_h_o > 1) ? h : 0;
                tmpA_pos_h = ah*(A_i*A_j*A_m*A_k);
                tmpB_pos_h = bh*(B_i*B_j*B_k*B_n);
                tmpY_pos_h =  h*(Y_i*Y_j*Y_m*Y_n);
                for (i=0; i < Y_i; i++) {{
                    ai = (A_i_o > 1) ? i : 0;
                    bi = (B_i_o > 1) ? i : 0;
                    tmpA_pos_i = tmpA_pos_h + ai*(A_j*A_m*A_k);
                    tmpB_pos_i = tmpB_pos_h + bi*(B_j*B_k*B_n);
                    tmpY_pos_i = tmpY_pos_h +  i*(Y_j*Y_m*Y_n);
                    for (j=0; j < Y_j; j++) {{
                        aj =  (A_j_o > 1) ? j : 0;
                        bj =  (B_j_o > 1) ? j : 0;
                        tmpA_pos = tmpA_pos_i + aj*(A_m*A_k);
                        tmpB_pos = tmpB_pos_i + bj*(B_k*B_n);
                        tmpY_pos = tmpY_pos_i +  j*(Y_m*Y_n);
#pragma omp parallel for private(m,n,k,multiplier,tmpA,tmpB) reduction(+:tmpY)
                        for (m=0; m < Y_m; m++) {{
                            for (n=0; n < Y_n; n++) {{
                                tmpY = 0;
                                multiplier = a_scale_mod[m] * b_scale_mod[n] / y_scale_mod[m];
                                for (k=0; k < B_k; k++) {{
                                    tmpA = *(_A  + tmpA_pos + m*(A_k) + k) - a_zero_point_mod[m];
                                    tmpB = *(_BT + tmpB_pos + n*(B_k) + k) - b_zero_point_mod[n];
                                    tmpY += tmpA * tmpB;
                                }}
                                *(_Y + tmpY_pos + m*(Y_n) + n) = qlinearmatmul_CLAMP(qlinearmatmul_ROUND(multiplier * tmpY + y_zero_point_mod[m]),{Y_min},{Y_max});
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
        mapping.update({'ta': data_type.np2c(self.input_tensor_dtypes[0])})
        mapping.update({'tb': data_type.np2c(self.input_tensor_dtypes[3])})
        mapping.update({'ty': data_type.np2c(self.output_tensor_dtypes[0])})
        mapping.update({'taz': data_type.np2c(self.input_tensor_dtypes[2])})
        mapping.update({'tbz': data_type.np2c(self.input_tensor_dtypes[5])})
        mapping.update({'tyz': data_type.np2c(self.input_tensor_dtypes[7])})
        mapping.update({'A_d0_o': input_org_shapes[0][0]})
        mapping.update({'A_d1_o': input_org_shapes[0][1]})
        mapping.update({'A_d2_o': input_org_shapes[0][2]})
        mapping.update({'B_d0_o': input_org_shapes[1][0]})
        mapping.update({'B_d1_o': input_org_shapes[1][1]})
        mapping.update({'B_d2_o': input_org_shapes[1][2]})
        mapping.update({'Y_min': 0 if (self.output_tensor_dtypes[0] == 'uint8') else -128})
        mapping.update({'Y_max': 255 if (self.output_tensor_dtypes[0] == 'uint8') else 127})


        # 3
        TemplateFunction = cleandoc('''
            void {op_func_name}(void *op_param, {ta} A{dims_A}, float a_scale[{dims_az}], {taz} a_zero_point[{dims_az}], {tb} B{dims_B}, float b_scale[{dims_bz}], {tbz} b_zero_point[{dims_bz}], float y_scale[{dims_yz}], {tyz} y_zero_point[{dims_yz}], {ty} Y{dims}, void *inputs_params, void* outputs_params)
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
        mappingf.update({'ta': data_type.np2c(self.input_tensor_dtypes[0])})
        mappingf.update({'tb': data_type.np2c(self.input_tensor_dtypes[3])})
        mappingf.update({'ty': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'dims_az': zero_point_shapes[0]})
        mappingf.update({'dims_bz': zero_point_shapes[1]})
        mappingf.update({'dims_yz': zero_point_shapes[2]})
        mappingf.update({'taz': data_type.np2c(self.input_tensor_dtypes[2])})
        mappingf.update({'tbz': data_type.np2c(self.input_tensor_dtypes[5])})
        mappingf.update({'tyz': data_type.np2c(self.input_tensor_dtypes[7])})
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

    """  
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
    """

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


