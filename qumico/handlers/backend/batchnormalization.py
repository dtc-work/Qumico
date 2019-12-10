from inspect import cleandoc
from collections import OrderedDict

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op('BatchNormalization')

class BatchNormalization(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]

        attrs = node.attrs
        attrs.update({'epsilon': attrs.get('epsilon', 1e-5)})
        attrs.update({'momentum': attrs.get('momentum', 0.9)})
        if (cls.SINCE_VERSION >= 9):
            if (attrs.get('spatial') != None):
                raise(ValueError)
        attrs.update({'spatial': attrs.get('spatial', 1)})
        outputs_shape = input_data1.shape
        outputs_dtype = input_data1.dtype
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return 'BatchNormalizationOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['batchnormalization.c']


    @classmethod
    def get_c_op_include_header(cls):
        return ['math.h']
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        return cleandoc(
            '''
            typedef struct {
                char* name;
                double epsilon;
                double momentum;
                int    spatial;
            } BatchNormalizationOpParam;
            ''')


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        TemplateStatements = '''
            const int Y_n = {d1};
            const int Y_c = {d2};
            const int Y_h = {d3};
            const int Y_w = {d4};

            const double epsilon =  {epsilon};
            const double momentum = {momentum};
            const int    spatial =  {spatial};

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if {spatial} // spatial is true
            for (n=0; n<Y_n; n++) {{
                for (c=0; c<Y_c; c++) {{
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            sum += X[n][c][h][w];
                        }}
                    }}
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }}
                    }}
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }}
                    }}
                }}
            }}
#else // spatial is false
            for (n=0; n<Y_n; n++) {{
                for (c=0; c<Y_c; c++) {{
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            sum += X[n][c][h][w];
                        }}
                    }}
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }}
                    }}
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }}
                    }}
                }}
            }}
#endif // spatial
        '''

        mapping = {}
        mapping.update({'d1': self.input_tensor_shapes[0][0]})
        mapping.update({'d2': self.input_tensor_shapes[0][1]})
        mapping.update({'d3': self.input_tensor_shapes[0][2]})
        mapping.update({'d4': self.input_tensor_shapes[0][3]})
        mapping.update({'epsilon': self.attrs['epsilon']})
        mapping.update({'momentum': self.attrs['momentum']})
        mapping.update({'spatial': self.attrs['spatial']})

        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} X{dims_X}, {t} scale{dims_scale}, {t} B{dims_B}, {t} mean{dims_mean}, {t} var{dims_var}, {t} Y{dims}, void *inputs_params, void* outputs_params) {{
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'dims_X': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({'dims_scale': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])}) 
        mappingf.update({'dims_B': c_helper.generate_dim_bracket(self.input_tensor_shapes[2])}) 
        mappingf.update({'dims_mean': c_helper.generate_dim_bracket(self.input_tensor_shapes[3])}) 
        mappingf.update({'dims_var': c_helper.generate_dim_bracket(self.input_tensor_shapes[4])}) 
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
    def version_7(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
