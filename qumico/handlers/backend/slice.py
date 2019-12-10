import string
from inspect import cleandoc
from itertools import zip_longest
import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type


@onnx_op('Slice')
class Slice(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        # parms
        n_starts = node.attrs.get('starts') # required
        n_ends = node.attrs.get('ends')     # required
        n_slice_len = len(n_starts)           
        n_axes = node.attrs.get('axes', list(range(n_slice_len))) # optional 
        input = node.input_tensor[0]

        c_starts =[]
        c_ends =[] 
        c_axis_mask = []
        c_shape = []
        iter_n_starts = iter(n_starts) 
        iter_n_ends = iter(n_ends)
        for i, s in enumerate(input.shape):
            if i in n_axes:
                c_starts.append(next(iter_n_starts))
                end = next(iter_n_ends)   
                end =  s if s <= end else end  
                c_ends.append(s if s <= end else end)
                c_axis_mask.append(True)
            else:
                c_starts.append(0)
                c_ends.append(s)
                c_axis_mask.append(False)
            c_shape.append(c_ends[-1] - c_starts[-1])

        output =np.array(input, copy=True)
        for s,e, d in zip(c_starts, c_ends, range(input.ndim)):
            output = np.take(output, range(s,e), d)

        output_value ={node.valid_var_name(node.outputs[0]): output}
        output_tensor = namedtupledict('output_tensor', output_value.keys())(**output_value)

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor, attrs=node.attrs,
                   c_axis_mask=c_axis_mask,c_starts=c_starts, c_ends=c_ends)
    

    def __init__(self, *args, **kwargs):
        super(Slice, self).__init__(*args, **kwargs)
        self.c_starts = kwargs.get('c_starts')
        self.c_ends = kwargs.get('c_ends')
        self.c_axis_mask = kwargs.get('c_axis_mask')
    
    @classmethod
    def get_param_type_name(cls):
        return   'SliceOpParam'    

    @classmethod
    def get_c_op_file_name(cls):
        return ['slice.c']

    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
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
            } SliceOpParam;
            ''')

    def generate_c_code(self, **kwargs):
        res =''
        
        # include header
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'
        
        ndim = len(self.c_starts)
        indent = [' ' * 4] * ndim
        alpha_iter = reversed(string.ascii_lowercase[8:8+ndim])
        TemplateArrayLoop = ''.join(indent) + '[statements]'
        input_dim = ''
        output_dim = ''
        for s, e in zip(self.c_starts[::-1], self.c_ends[::-1]):
            var = next(alpha_iter)
            params = {}
            params.update({'var': var})
            params.update({'start': str(0)})
            params.update({'end': str(e)})

            loop_start =''.join(indent) + 'for(int {var}={start};{var}<{end};{var}++ ){{'.format(**params ) 
            loop_end = ''.join(indent) +'}'
            TemplateArrayLoop =loop_start  +'\n' +  TemplateArrayLoop + '\n' + loop_end
            
            input_dim = '[' + str(var) + ('' if s==0 else '+' + str(s)) + ']' + input_dim 
            output_dim = '[' + str(var) + ']' + output_dim
            indent.pop()       

        statements = TemplateArrayLoop.replace('[statements]',  ''.join(indent) + 'output' + output_dim + '=' + 'input' + input_dim + ';')
        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param,{t} input{XDims}, {t} output{CDims}, void *inputs_params, void* outputs_params)
        {{
        {statements}
        }}
        ''')
        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'XDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})
        mappingf.update({'CDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})
        mappingf.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'statements':statements})
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


    @classmethod
    def version_10(cls, node, **kwargs):
        raise NotImplementedError()