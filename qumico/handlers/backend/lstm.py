from collections import OrderedDict
from enum import Enum
from inspect import cleandoc

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.common import c_helper, data_type
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op



class Direction(Enum):
    forward = "forward"
    reverse = "reverse"
    bidirectional = "bidirectional"


class LSTMActivate(Enum):
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


@onnx_op('LSTM')
class LSTM(BackendHandler):
    
    @classmethod
    def instantiate(cls, node, **kwargs):

        cls.activates = []
        clip = node.attrs.get('clip', None)   # float
        
        # string (default is forward)
        hidden_size = node.attrs['hidden_size']  # int (required)
        # ToDo: the bellows are not supported attributes
        direction = node.attrs.get('direction', Direction.forward) 
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

        if not num_directions ==1:
            raise ValueError("Not Supported: BiDirectional LSTM") 

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
        return   'LSTMOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['lstm.c']


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
            } LSTMOpParam;
            ''')


    @classmethod
    def get_activate_func(cls,func):

        if func not in cls.activates and func==LSTMActivate.Sigmoid.name:
            mapping = {}
            mapping.update({"func":LSTMActivate.Sigmoid.value})
            
            cls.activates.append(LSTMActivate.Sigmoid.name)

            return cleandoc("""
            float {func}(float x, float alpha, float beta){{
                return 1 / (1 + exp(-x));
            }}
            """).format(**mapping)

        if func not in cls.activates and func==LSTMActivate.HardSigmoid.name:
            mapping = {}
            mapping.update({"func":LSTMActivate.HardSigmoid.value})

            cls.activates.append(LSTMActivate.HardSigmoid.name)

            return cleandoc("""
            float {func}(float x, float alpha, float beta){{
                return fminf(fmaxf(alpha*x + beta, 0), 1);
             }}
            """).format(**mapping)

        if func not in cls.activates and func==LSTMActivate.Tanh.name: 
            mapping = {}
            mapping.update({"func":LSTMActivate.Tanh.value})

            cls.activates.append(LSTMActivate.Tanh.name)

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
        mapping.update({'X': "vi_X"})
        mapping.update({'XDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[0])})

        mapping.update({'wt': data_type.np2c(self.input_tensor_dtypes[1])})
        mapping.update({'W': self.input_tensor_names[1]})
        mapping.update({'WDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[1])})

        mapping.update({'rt': data_type.np2c(self.input_tensor_dtypes[2])})
        mapping.update({'R': self.input_tensor_names[2]})
        mapping.update({'RDims':c_helper.generate_dim_bracket(self.input_tensor_shapes[2])})


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

        if 6 < len(self.input_tensor):
            mapping.update({'initial_ct': data_type.np2c(self.input_tensor_dtypes[6])})
            mapping.update({'initial_c': self.input_tensor_names[6]})
            mapping.update({'initial_cDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[6])})
            res += "{initial_ct} {initial_c}{initial_cDims},"

        if 7 < len(self.input_tensor):
            mapping.update({'pt': data_type.np2c(self.input_tensor_dtypes[7])})
            mapping.update({'P': self.input_tensor_names[7]})
            mapping.update({'PDims': c_helper.generate_dim_bracket(self.input_tensor_shapes[7])})
            res += "{pt} {P}{PDims},"
            
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

        res += " void *inputs_params, void* outputs_params)"

        return res.format(**mapping)


    def gen_c_bias(self, b_half_index, number_of_gates=4):
        res = ""
        if 3< len(self.input_tensor):
            res = """
                ti[j][k] +=(vi_B[0][k+0*hidden_size] + vi_B[0][{b_half_index} + k + 0*hidden_size]);
                to[j][k] +=(vi_B[0][k+1*hidden_size] + vi_B[0][{b_half_index} + k + 1*hidden_size]);
                tf[j][k] +=(vi_B[0][k+2*hidden_size] + vi_B[0][{b_half_index} + k + 2*hidden_size]);
                tc[j][k] +=(vi_B[0][k+3*hidden_size] + vi_B[0][{b_half_index} + k + 3*hidden_size]);
            """.format(**{"number_of_gates":number_of_gates, "b_half_index":b_half_index})

        return res


    def generate_c_code(self, **kwargs):
        hidden_size = self.attrs['hidden_size']
        batch_size = self.input_tensor_shapes[0][1]
        input_size = self.input_tensor_shapes[0][-1]
        sequence_lens = self.input_tensor_shapes[0][0]
        num_directions = self.input_tensor_shapes[1][0]
        b_half_index = self.input_tensor_shapes[3][-1] // 2 if 3 < len(self.input_tensor) else 0

        HDim = [sequence_lens,num_directions, batch_size, hidden_size]
        
        res =''

        # include header
        res += '\n'.join([c_helper.generate_std_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n\n'
        res += ""
            
        # activate func
        activations =  ['Sigmoid','Tanh', 'Tanh']   # default
        activation_alpha = ["0", "0", "0"]          # default
        activation_beta =  ["0", "0", "0"]          # default

        func = list(self.attrs.get('activations',[]))
        alpha = list(self.attrs.get('activation_alpha',[]))
        beta = list(self.attrs.get('activation_beta',[]))

        activations[:len(func)] = func
        activation_alpha[:len(alpha)] = alpha 
        activation_beta[:len(beta)] = beta

        for act in set(activations):    
            res += self.get_activate_func(act) + '\n\n'

        res +='\n\n'
                
        mappingf = {}
        mappingf.update({"signature":self.get_signature()})

        mappingf.update({'input_size': input_size})
        mappingf.update({'batch_size': batch_size})
        mappingf.update({'hidden_size': hidden_size})
        mappingf.update({'sequence_lens': sequence_lens})

        mappingf.update({"PLen": str(3 * hidden_size)})
        mappingf.update({"P": self.input_tensor_names[7] + "[0]" if len(self.input_tensor)==8 else "P"})

        mappingf.update({"HDim_last2": c_helper.generate_dim_bracket(HDim[-2:])})
        mappingf.update({"bias_code": self.gen_c_bias(b_half_index)})

        # activate
        mappingf.update({"act_f": LSTMActivate[activations[0]].value})
        mappingf.update({"act_g": LSTMActivate[activations[1]].value})
        mappingf.update({"act_h": LSTMActivate[activations[2]].value})

        mappingf.update({"act_alpha_f": activation_alpha[0]})
        mappingf.update({"act_alpha_g": activation_alpha[1]})
        mappingf.update({"act_alpha_h": activation_alpha[2]})

        mappingf.update({"act_beta_f": activation_beta[0]})
        mappingf.update({"act_beta_g": activation_beta[1]})
        mappingf.update({"act_beta_h": activation_beta[2]})

        mappingf.update({"Y_set_code": ""})
        mappingf.update({"Y_h_set_code": ""})
        mappingf.update({"Y_c_set_code": ""})
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

            elif index==2 and  self.node.outputs[index] != "":# Y_c exist:
                mappingf.update({'yt': data_type.np2c(self.output_tensor_dtypes[mappingf_index])})
                mappingf.update({'Y_c': "vi_Y_c"})
                mappingf.update({'Y_cDims': c_helper.generate_dim_bracket(self.output_tensor_shapes[mappingf_index])})
                mappingf.update({"Y_c_set_code": "memcpy(&vi_Y_c[0], &(tc), sizeof(tc));"})# only support OneDirectino
        

        TemplateFunction = cleandoc('''
        {signature}
        {{
            const int hidden_size={hidden_size};
            const {yt} P[{PLen}] ={{0.0}};

            {yt} prevH{HDim_last2} = {{0.0}};
            {yt} prevC{HDim_last2} = {{0.0}};

            for(int i =0;i<{sequence_lens};i++)
            {{
                float y[{batch_size}][{hidden_size}] = {{0.0}};
                float ti[{batch_size}][{hidden_size}] = {{0.0}};
                float to[{batch_size}][{hidden_size}] = {{0.0}};
                float tf[{batch_size}][{hidden_size}] = {{0.0}};
                float tc[{batch_size}][{hidden_size}] = {{0.0}};            

                for(int j=0;j<{batch_size};j++){{
                    for(int k=0;k<{hidden_size};k++){{

                        for(int l=0;l<{input_size};l++){{
                            ti[j][k] +=(vi_X[i][j][l] * vi_W[0][k + hidden_size * 0][l]);
                            to[j][k] +=(vi_X[i][j][l] * vi_W[0][k + hidden_size * 1][l]);
                            tf[j][k] +=(vi_X[i][j][l] * vi_W[0][k + hidden_size * 2][l]);
                            tc[j][k] +=(vi_X[i][j][l] * vi_W[0][k + hidden_size * 3][l]);
                        }}
        
                        for(int l=0;l<{hidden_size}; l++){{
                            ti[j][k] +=(prevH[j][l] * vi_R[0][k + hidden_size * 0][l]);
                            to[j][k] +=(prevH[j][l] * vi_R[0][k + hidden_size * 1][l]);
                            tf[j][k] +=(prevH[j][l] * vi_R[0][k + hidden_size * 2][l]);
                            tc[j][k] +=(prevH[j][l] * vi_R[0][k + hidden_size * 3][l]);
                        }}
                        {bias_code}
                    }}
                }}

                for(int j=0;j<{batch_size};j++){{
                    for(int k=0;k<{hidden_size};k++){{
                        ti[j][k] = {act_f}(ti[j][k] + {P}[k + 3 * 0] * prevC[j][k], {act_alpha_f}, {act_beta_f});
                        tf[j][k] = {act_f}(tf[j][k] + {P}[k + 3 * 1] * prevC[j][k], {act_alpha_f}, {act_beta_f});
                        tc[j][k] = {act_g}(tc[j][k], {act_alpha_g}, {act_beta_g});
                        tc[j][k] = tf[j][k] * prevC[j][k] + ti[j][k] * tc[j][k];
                        to[j][k] = {act_f}(to[j][k] + {P}[k + 3 * 2]* tc[j][k], {act_alpha_f}, {act_beta_f});
                        y[j][k] = to[j][k] * {act_h}(tc[j][k], {act_alpha_h}, {act_beta_h});
                    }}
                }}

                {Y_set_code}
                {Y_h_set_code}
                {Y_c_set_code}
                memcpy(&prevH, &(y), sizeof(y));
                memcpy(&prevC, &(tc), sizeof(tc));

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
