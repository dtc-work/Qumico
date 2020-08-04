import string
from inspect import cleandoc
from collections import OrderedDict

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op('Gemm')
class Gemm(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        attr_transA = node.attrs.get('transA', 0)
        attr_transB = node.attrs.get('transB', 0)
        input_data1 = node.input_tensor[0]
        input_data2 = node.input_tensor[1]
        input_data3 = node.input_tensor[2]

        if attr_transA == 0:
            m = input_data1.shape[0]
        else:
            m = input_data1.shape[1]
        if attr_transB == 0:
            n = input_data2.shape[1]
        else:
            n = input_data2.shape[0]

        node.attrs.update({'transA': attr_transA})
        node.attrs.update({'transB': attr_transB})
        node.attrs.update({'alpha': node.attrs.get('alpha', 1.0)})
        node.attrs.update({'beta':  node.attrs.get('beta', 1.0)})
        if (input_data3.ndim <= 1):
            input_data3 = np.reshape(input_data3, (1, input_data3.shape[0]))
        outputs_shape = (max(m, input_data3.shape[0]), max(n, input_data3.shape[1]))
        outputs_dtype = input_data1.dtype if input_data1.dtype == input_data2.dtype else np.double
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=node.attrs)

    @classmethod
    def get_param_type_name(cls):
        return 'GemmOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['gemm.c']


    @classmethod
    def get_c_op_include_header(cls):
        return ['string.h']
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
                {t} alpha;
                {t} beta;
                int transA;
                int transB;
            }} GemmOpParam;
            '''
        )
        mapping = {
            't': 'double'
        }

        return TEMPLATE_STRUCT.format(**mapping)


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        attr_alpha = self.attrs['alpha']
        attr_beta = self.attrs['beta']
        attr_transA = self.attrs['transA']
        attr_transB = self.attrs['transB']

        TemplateStatements = '''
            GemmOpParam *param_ptr = (GemmOpParam *)op_param;
            {t} alpha = {alpha};    //param_ptr->alpha
            {t} beta = {beta};      //param_ptr->beta
            int transA = {transA};  //param_ptr->transA;
            int transB = {transB};  //param_ptr->transB;

            {t}  *_C = ({t} *)(C);
            int   A_m;
            int   A_k;
            int   B_k;
            int   B_n;
            int   C_m = {C_d0};
            int   C_n = {C_d1};
            int   Y_m = {Y_d0};
            int   Y_n = {Y_d1};
  
            int   k;
            int   m;
            int   n;
            int   tmp_m, tmp_n;

            if (transA == 0) {{
                A_m = {A_d0};
                A_k = {A_d1};
            }} else {{
                A_m = {A_d1};
                A_k = {A_d0};
            }}
            if (transB == 0) {{
                B_k = {B_d0};
                B_n = {B_d1};
            }} else {{
                B_k = {B_d1};
                B_n = {B_d0};
            }}

            // extend dimension of C if C is 1D array.
            if (C_n == 0) {{
                C_n = 1;
            }}
            memset( Y, ({t})0.0, sizeof(Y[0][0])*Y_m*Y_n );

            for (m=0; m < Y_m; m++) {{
                for (n=0; n < Y_n; n++) {{
                    for (k=0; k < B_k; k++) {{
                        if (transA ==0 && transB == 0) {{
                            Y[m][n] += A[m][k] * B[k][n];
                        }} else if (transA ==0 && transB == 1) {{
                            Y[m][n] += A[m][k] * B[n][k];
                        }} else if (transA ==1 && transB == 0) {{
                            Y[m][n] += A[k][m] * B[k][n];
                        }} else if (transA ==1 && transB == 1) {{
                            Y[m][n] += A[k][m] * B[n][k];
                        }}
                    }}
                    Y[m][n] *= alpha;
#if {broadcast} // BROADCAST
                    if (m >= C_m) {{
                        tmp_m = m % C_m;
                    }} else {{
                        tmp_m = m;
                    }}
                    if (n >= C_n) {{
                        tmp_n = n % C_n;
                    }} else {{
                        tmp_n = n;
                    }}
#else // BROADCAST
                    tmp_m = m;
                    tmp_n = n;
#endif // BROADCAST
                    Y[m][n]  += beta * *(_C + tmp_m*C_n + tmp_n);
                }}
            }}
        '''

        mapping = {}
        mapping.update({'A': self.input_tensor_names[0]})
        mapping.update({'A_d0': self.input_tensor_shapes[0][0]})
        mapping.update({'A_d1': self.input_tensor_shapes[0][1]})
        mapping.update({'B': self.input_tensor_names[1]})
        mapping.update({'B_d0': self.input_tensor_shapes[1][0]})
        mapping.update({'B_d1': self.input_tensor_shapes[1][1]})
        mapping.update({'C': self.input_tensor_names[2]})
        if (self.input_tensor_ndims[2] <= 1):
            mapping.update({'C_d0': 1})
            mapping.update({'C_d1': self.input_tensor_shapes[2][0]})
        else:
            mapping.update({'C_d0': self.input_tensor_shapes[2][0]})
            mapping.update({'C_d1': self.input_tensor_shapes[2][1]})
        mapping.update({'Y': self.output_tensor_names[0]})
        mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
        mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
        mapping.update({'alpha': attr_alpha})
        mapping.update({'beta': attr_beta})
        mapping.update({'transA': attr_transA})
        mapping.update({'transB': attr_transB})
        mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        if (self.input_tensor_ndims[2] <= 1):
            broadcast = 1
        elif ((self.input_tensor_shapes[2][0] >= self.output_tensor_shapes[0][0])
            and (self.input_tensor_shapes[2][1] >= self.output_tensor_shapes[0][1])):
            broadcast = 0
        else:
            broadcast = 1
        mapping.update({'broadcast': broadcast})

        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} A{dims_A}, {t} B{dims_B}, {t} C{dims_C}, {t} Y{dims}, void *inputs_params, void* outputs_params)
        {{
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'A': self.input_tensor_names[0]})
        mappingf.update({'dims_A': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({'B': self.input_tensor_names[1]})
        mappingf.update({'dims_B': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])}) 
        mappingf.update({'C': self.input_tensor_names[2]})
        mappingf.update({'dims_C': c_helper.generate_dim_bracket(self.input_tensor_shapes[2])}) 
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
        ndim = self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]
        mapping = {}
        mapping.update({'shape': ','.join(map(str,shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc('''
        {indent}// define input & output
        {indent}{node_param_name}.alpha = {alpha};
        {indent}{node_param_name}.beta = {beta};
        {indent}{node_param_name}.transA = {transA};
        {indent}{node_param_name}.transB = {transB};
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
        mapping.update({'beta': node.attrs['beta']})
        mapping.update({'transA': node.attrs['transA']})
        mapping.update({'transB': node.attrs['transB']})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def version_7(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)



