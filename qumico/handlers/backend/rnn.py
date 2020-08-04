from collections import OrderedDict
from enum import Enum
from inspect import cleandoc

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type


class Direction(Enum):
    forward = "forward"
    reverse = "reverse"
    bidirectional = "bidirectional"


class RNNActivate(Enum):
    Relu = "activate_relu"
    Tanh = "activate_tanh"
    Sigmoid = "activate_sigmoid"
    Affine = "activate_affine"
    LeakyRelu = "activate_leaky_relu"
    ThresholdedRelu = "activate_thresholded_relu"
    ScaledTanh = "activate_scaled_tanh"
    HardSigmoid = "activate_hard_sigmoid"
    Elu = "activate_elu"
    Softsign = "activate_softsign"
    Softplus = "activate_softplus"



@onnx_op('RNN')
class RNN(BackendHandler):
    
    @classmethod
    def instantiate(cls, node, **kwargs):
        cls.activates = []
        # attribute
        clip = node.attrs.get('clip', None)   # float
        
        # string (default is forward)
        direction = node.attrs.get('direction', Direction.forward) 
        hidden_size = node.attrs['hidden_size']  # int (required)        
        linear_before_reset = node.attrs.get("linear_before_reset", 0) # int (default is 0)

        # input        
        x = node.input_tensor[0] # requied
        w = node.input_tensor[1] # requied
        r = node.input_tensor[2] # requied

        seq_length = x.shape[0]
        num_directions = w.shape[0]
        batch_size =  x.shape[1]

        assert num_directions  == w.shape[0]
        assert num_directions  == r.shape[0]
        # B
        if 4 < len(node.input_tensor):
            raise NotImplementedError("optional input not supported")


        output_dict = OrderedDict()

        for index, _ in enumerate(node.outputs):

            if index==0 and node.outputs[index] != "": 
                output_dict.update({node.valid_var_name(node.outputs[index]):
                                    np.ones(shape=[seq_length, num_directions, batch_size, hidden_size],
                                            dtype=x.dtype)}) 
            elif node.outputs[index] != "":
                output_dict.update({node.valid_var_name(node.outputs[index]):
                                    np.ones(shape=[num_directions, batch_size, hidden_size],
                                            dtype=x.dtype)})

        output_tensor = namedtupledict('output_tensor', output_dict.keys())(**output_dict)

        return cls(node, input_tensor=node.input_tensor,
                   output_tensor=output_tensor,attrs=node.attrs,  **kwargs)
    

    @classmethod
    def get_param_type_name(cls):
        return   'RNNOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['rnn.c']


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ["math.h", "string.h", "stdio.h"]


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
            } RNNOpParam;
            ''')

    @classmethod
    def get_activate_func(cls,func):

        if func not in cls.activates and func==RNNActivate.Sigmoid.name:
            mapping = {}
            mapping.update({"func":RNNActivate.Sigmoid.value})
            
            cls.activates.append(RNNActivate.Sigmoid.name)

            return cleandoc("""
            float {func}(float x, float alpha, float beta){{
                return 1 / (1 + exp(-x));
            }}
            """).format(**mapping)

        if func not in cls.activates and func==RNNActivate.HardSigmoid.name:
            mapping = {}
            mapping.update({"func":RNNActivate.HardSigmoid.value})

            cls.activates.append(RNNActivate.HardSigmoid.name)

            return cleandoc("""
            float {func}(float x, float alpha, float beta){{
                return fminf(fmaxf(alpha*x + beta, 0), 1);
             }}
            """).format(**mapping)

        if func not in cls.activates and func==RNNActivate.Tanh.name: 
            mapping = {}
            mapping.update({"func":RNNActivate.Tanh.value})

            cls.activates.append(RNNActivate.Tanh.name)

            return cleandoc("""
            float {func}(float x, float alpha, float beta){{
                return tanh(x);
             }}
            """).format(**mapping)

        else:
            raise ValueError("Unspported Activate Func")


    def get_signature(self):

        res = "void {op_func_name}(void *op_param,{xt} {X}{XDims},{wt} {W}{WDims},{rt} {R}{RDims},"
        mapping = {}
        mapping.update({'op_func_name': self.get_func_name()})
        mapping.update({'xt': data_type.np2c(self.input_tensor_dtypes[0])})
        mapping.update({'X': self.input_tensor_names[0]})
        mapping.update({'XDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})

        mapping.update({'wt': data_type.np2c(self.input_tensor_dtypes[1])})
        mapping.update({'W': self.input_tensor_names[1]})
        mapping.update({'WDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[1])})

        mapping.update({'rt': data_type.np2c(self.input_tensor_dtypes[2])})
        mapping.update({'R': self.input_tensor_names[2]})
        mapping.update({'RDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[2])})
        
        mapping.update({'yt': data_type.np2c(self.output_tensor_dtypes[0])})
        mapping.update({'Y': self.output_tensor_names[0]})
        mapping.update({'YDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[0])})

        if 3 < len(self.input_tensor):
            mapping.update({'bt': data_type.np2c(self.input_tensor_dtypes[3])})
            mapping.update({'B': self.input_tensor_names[3]})
            mapping.update({'BDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[3])})
            res += "{bt} {B}{BDims},"

        if 4 < len(self.input_tensor):
            mapping.update({'slt': data_type.np2c(self.input_tensor_dtypes[4])})
            mapping.update({'sl': self.input_tensor_names[4]})
            mapping.update({'slDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[4])})
            res += "{slt} {sl}{slDims},"

        if 5 < len(self.input_tensor):
            mapping.update({'initial_ht': data_type.np2c(self.input_tensor_dtypes[5])})
            mapping.update({'initial_h': self.input_tensor_names[5]})
            mapping.update({'initial_hDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[5])})
            res += "{initial_ht} {initial_h}{initial_hDims},"
            
        # output
        mapping_index =0
        for index, o in enumerate(range(len(self.node.outputs))):# onnx definitive order
            if index==0 and self.node.outputs[index] != "":# Y exist
                mapping.update({'yt': data_type.np2c(self.output_tensor_dtypes[mapping_index])})
                mapping.update({'Y': "vi_Y"})
                mapping.update({'YDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mapping_index])})

                mapping_index +=1

                res += "{yt} {Y}{YDims},"

            elif index==1 and  self.node.outputs[index] != "":# Y_h exist
                mapping.update({'y_ht': data_type.np2c(self.output_tensor_dtypes[mapping_index])})
                mapping.update({'Y_h': "vi_Y_h"})
                mapping.update({'Y_hDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mapping_index])})

                mapping_index +=1

                res += "{y_ht} {Y_h}{Y_hDims},"

            elif index==2 and  self.node.outputs[index] != "":# Y_c exist:
                mapping.update({'y_ct': data_type.np2c(self.output_tensor_dtypes[mapping_index])})
                mapping.update({'Y_c': "vi_Y_c"})
                mapping.update({'Y_cDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mapping_index])})

                res += "{y_ct} {Y_c}{Y_cDims},"
        
        res += "void *inputs_params, void* outputs_params)"

        return res.format(**mapping)


    def gen_c_bias(self):
        res = ""
        if 3 < len(self.input_tensor):
            res = """
                y[j][k] += ({B}[0][k] + {B}[0][k + hidden_size]);
            """.format(**{"Y":self.output_tensor_names[0], "B": self.input_tensor_names[3]})

        return res


    def generate_c_code(self, **kwargs):
        hidden_size = self.attrs['hidden_size']
        batch_size = self.input_tensor_shapes[0][1]
        input_size = self.input_tensor_shapes[0][-1]
        sequence_lens = self.input_tensor_shapes[0][0]
        HDim = [sequence_lens] + list(self.output_tensor_shapes[0][1:]) 

        # activate func
        activations =  ['Tanh']   # default
        activation_alpha = ["0"]      # default
        activation_beta =  ["0"]       # default
        func = list(self.attrs.get('activations',[]))
        alpha = list(self.attrs.get('activation_alpha',[]))
        beta = list(self.attrs.get('activation_beta',[]))

        activations[:len(func)] = func
        activation_alpha[:len(alpha)] = alpha 
        activation_beta[:len(beta)] = beta

        res =''

        # include header
        res += '\n'.join([c_helper.generate_std_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n\n'

        # activate func
        for act in set(activations):    
            res += self.get_activate_func(act) + '\n\n'

            
        mappingf = {}
        mappingf.update({"signature":self.get_signature()})

        mappingf.update({'input_size': input_size})
        mappingf.update({'batch_size': batch_size})
        mappingf.update({'hidden_size': hidden_size})
        mappingf.update({'sequence_lens': sequence_lens})

        mappingf.update({"HDim": c_helper.generate_dim_bracket(HDim)})
        mappingf.update({"HDim_last2": c_helper.generate_dim_bracket(HDim[-2:])})

        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'W': self.input_tensor_names[1]})
        mappingf.update({'R': self.input_tensor_names[2]})        
        mappingf.update({'yt': data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({'Y': self.output_tensor_names[0]})

        mappingf.update({"act_f": RNNActivate[activations[0]].value})
        mappingf.update({"act_alpha_f": activation_alpha[0]})
        mappingf.update({"act_beta_f": activation_beta[0]})
        
        mappingf.update({"Y_set_code": ""})
        mappingf.update({"Y_h_set_code": ""})
        # output
        mappingf_index=0
        for index, _ in enumerate(range(len(self.node.outputs))):# onnx definitive order
            if index==0 and self.node.outputs[index] != "":# Y exist
                mappingf.update({'yt': data_type.np2c(self.output_tensor_dtypes[mappingf_index])})
                mappingf.update({'Y': "vi_Y"})
                mappingf.update({'YDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mappingf_index])})
                mappingf.update({"Y_set_code": "memcpy(&vi_Y[i], &(y), sizeof(y));"})
                mappingf_index +=1

            elif index==1 and  self.node.outputs[index] != "":# Y_h exist
                mappingf.update({'yt': data_type.np2c(self.output_tensor_dtypes[mappingf_index])})
                mappingf.update({'Y_h': "vi_Y_h"})
                mappingf.update({'Y_hDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mappingf_index])})
                mappingf.update({"Y_h_set_code": "memcpy(&vi_Y_h[0], &(y), sizeof(y));"})# only support OneDirectino
                mappingf_index +=1

        
        res += '\n\n'

        mappingf.update({'bias_code': self.gen_c_bias()})

        TemplateFunction = cleandoc('''
        {signature}
        {{
            const int hidden_size={hidden_size};
            const int input_size = {input_size};

            {yt} prev{HDim_last2} = {{0.0}};

            for(int i =0;i<{sequence_lens};i++)
            {{
                {yt} y[{batch_size}][{hidden_size}] = {{0.0}};

                for(int j=0;j<{batch_size};j++){{
                    for(int k=0;k<{hidden_size};k++){{

                        for(int l=0;l<input_size;l++){{
                            y[j][k] += ({X}[i][j][l] * {W}[0][k][l]);
                        }}
        
                        for(int m=0;m<hidden_size;m++){{
                            y[j][k] += (prev[j][m] * {R}[0][k][m]);
                        }}
        
                        {bias_code}
                        y[j][k] = {act_f}(y[j][k], {act_alpha_f},{act_beta_f});

                    }}
                }}
                {Y_set_code}
                {Y_h_set_code}
                memcpy(&prev, &(y), sizeof(y));
            }}
        }}
        ''')
        
        res += TemplateFunction.format(**mappingf)

        return res


    def gen_op_variables(self, node, node_num, **kwargs):
        return ""    


    def gen_init_func(self, node, node_num, indent=4, **kwargs):
        return ""

  
    @classmethod
    def version_1(cls, node, **kwargs):
        raise NotImplementedError()

    
    @classmethod
    def version_7(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
