from inspect import cleandoc
import numpy as np

from onnx import numpy_helper
from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import data_type
from qumico.common import c_helper



@onnx_op('Constant')
class Constant(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
#         print(node.node_proto)
        attr_value = node.attrs['value']
        dtype = data_type.onnx2np(attr_value.data_type)
        value = numpy_helper.to_array(attr_value)
        output_value ={node.valid_var_name(node.outputs[0]):
                       np.ones(shape=value.shape, dtype=dtype)}
        output_tensor = namedtupledict('output_tensor', output_value.keys())(**output_value)
        return cls(node,input_tensor=node.input_tensor,
                   output_tensor=output_tensor, attrs=node.attrs)


    @classmethod
    def get_param_type_name(cls):
        return 'ConstantOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['constant.c']

    @classmethod
    def get_c_op_include_header(cls):
        return []
    

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
            } ConstantOpParam;
            ''')
         

    def _gen_c_code_value(self):
        TEMPALTE_CONSTANCT_VALUE = cleandoc(
            '''
            {t} {name}{dim_bracket} = {values};
            ''')

        attr_value = self.attrs['value']
        value = numpy_helper.to_array(attr_value)

        mapping = {'name': self.get_name(),
                   't':data_type.np2c(self.output_tensor_dtypes[0]),
                   'dim_bracket': c_helper.generate_dim_bracket(self.output_tensor_shapes[0]),
                   'values':c_helper.generate_c_array(value)}
        
        return TEMPALTE_CONSTANCT_VALUE.format(**mapping)


    def generate_c_code(self, **kwargs):
        TEMPALTE_CONSTANCT_FUNC = cleandoc('''
        void {op_func_name}(void *op_param, void *outputs, void* outputs_params){{
            ConstantOpParam *p = (ConstantOpParam *)op_param;
            int ndim;
            int* shape;
            void * value;
        
            ndim = p->ndim;
            shape = p->shape;
            value =({type} *) p->value;
            
            int len = 1;
            for(int i=0;i< ndim;i++){{
                len *=shape[i];
            }}
        
            memcpy(outputs, value, sizeof({type}) * len);
        }}
        ''')

        res = ''
        res += self.get_c_param_type() # call only once
        res += '\n\n\n'

        # constant value
        res += self._gen_c_code_value()
        res += '\n\n\n'

        # constant function
        mapping ={}
        mapping.update({'op_name': self.get_name()})
        mapping.update({'op_func_name': self.get_func_name()})
        mapping.update({'type': data_type.np2c(self.output_tensor_dtypes[0])})
        res += TEMPALTE_CONSTANCT_FUNC.format(**mapping)
        
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
        {indent}{node_param_name}.value =&{constant_name};
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
         {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        ''')

        mapping = {}
        mapping.update({'node_param_name': node.node_param_name})
        mapping.update({'node_num': str(node_num)})
        mapping.update({'constant_name': self.get_name()})
        mapping.update({'ndim':str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent':' ' * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)