from inspect import cleandoc
import math
import numpy as np

from onnx.backend.base import namedtupledict

from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import RaspberryPi3, QumicoDeviceType,QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op



@onnx_op('Conv')
class Conv(BackendHandler):

    PARALLEL_SIZE = 1
    WORK_PAD_I_SIZE =0
    WORK_PAD_W_SIZE =0
    WORK_PAD_O_SIZE =0

    SIMD = False
    OpenMP=False

    @classmethod
    def instantiate(cls, node, **kwargs):

        device = kwargs.get('device')
        if device.__class__ == RaspberryPi3 and QumicoDeviceType.ARMNeon in device.options:
            cls.PARALLEL_SIZE = device.PARALELL_SIZE_NEON
            cls.SIMD = True

        if (issubclass(device.__class__, QumicoDevice) and 
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True

        input_data1 = node.input_tensor[0]
        input_data2 = node.input_tensor[1]

        if (len(node.input_tensor) == 3):  # Bias
            input_data3 = node.input_tensor[2]
        attrs = node.attrs
        if (input_data1.ndim == 3):
            if (attrs.get('dilations') is None):
                attrs['dilations'] = (1,)
            if (attrs.get('group') is None):
                attrs['group'] = 1
            if (attrs.get('kernel_shape') is None):
                attrs['kernel_shape'] = (input_data2.shape[2],)
            if (attrs.get('strides') is None):
                attrs['strides'] = (1,)
            if (attrs.get('pads') is None):
                auto_pad = attrs.get('auto_pad')
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0,0)
#                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]+attrs['pads'][1])/attrs['strides'][0])+1)

            if cls.SIMD:
                work_pad_i = (node.input_tensor[0].shape[0] *
                              math.ceil(node.input_tensor[0].shape[1]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                             (node.input_tensor[0].shape[2] + attrs['pads'][0] + attrs['pads'][1])) 
#                work_pad_w = np.prod(node.input_tensor[1].shape)                
                work_pad_w = (node.input_tensor[1].shape[0] *
                              node.input_tensor[1].shape[1] *
                              math.ceil(node.input_tensor[1].shape[2]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE)
#                work_pad_o = np.prod(outputs_shape)
                work_pad_o = (input_data1.shape[0] *
                              math.ceil(input_data2.shape[0]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                             (math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]*2+attrs['pads'][1]*2)/attrs['strides'][0])+1))

                cls.WORK_PAD_I_SIZE = max(work_pad_i, cls.WORK_PAD_I_SIZE)
                cls.WORK_PAD_W_SIZE = max(work_pad_w, cls.WORK_PAD_W_SIZE)
                cls.WORK_PAD_O_SIZE = max(work_pad_o, cls.WORK_PAD_O_SIZE)
        
        elif (input_data1.ndim == 4):
            if (attrs.get('dilations') is None):
                attrs['dilations'] = (1, 1)
            if (attrs.get('group') is None):
                attrs['group'] = 1
            if (attrs.get('kernel_shape') is None):
                attrs['kernel_shape'] = (input_data2.shape[2], input_data2.shape[3])
            if (attrs.get('strides') is None):
                attrs['strides'] = (1, 1)
            if (attrs.get('pads') is None):
                auto_pad = attrs.get('auto_pad')
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0,0,0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0,0,0,0)
#                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]+attrs['pads'][2])/attrs['strides'][0])+1,
                math.floor((input_data1.shape[3]-1-(attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+attrs['pads'][1]+attrs['pads'][3])/attrs['strides'][1])+1)

            if cls.SIMD:
                work_pad_i = (node.input_tensor[0].shape[0] *
                              math.ceil(node.input_tensor[0].shape[1]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                            (node.input_tensor[0].shape[2] + attrs['pads'][0] + attrs['pads'][2]) * 
                            (node.input_tensor[0].shape[3] + attrs['pads'][1] + attrs['pads'][3]))
#                work_pad_w = np.prod(node.input_tensor[1].shape)
                work_pad_w = (node.input_tensor[1].shape[0] *
                              node.input_tensor[1].shape[1] *
                              math.ceil(node.input_tensor[1].shape[2]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                              math.ceil(node.input_tensor[1].shape[3]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE)
#                work_pad_o = np.prod(outputs_shape)
                work_pad_o = (input_data1.shape[0] *
                             math.ceil(input_data2.shape[0]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                             (math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]*2+attrs['pads'][2]*2)/attrs['strides'][0])+1) *
                             (math.floor((input_data1.shape[3]-1-(attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+attrs['pads'][1]*2+attrs['pads'][3]*2)/attrs['strides'][1])+1))

                cls.WORK_PAD_I_SIZE = max(work_pad_i, cls.WORK_PAD_I_SIZE)
                cls.WORK_PAD_W_SIZE = max(work_pad_w, cls.WORK_PAD_W_SIZE)
                cls.WORK_PAD_O_SIZE = max(work_pad_o, cls.WORK_PAD_O_SIZE)
        
        elif (input_data1.ndim == 5):
            if (attrs.get('dilations') is None):
                attrs['dilations'] = (1, 1, 1)
            if (attrs.get('group') is None):
                attrs['group'] = 1
            if (attrs.get('kernel_shape') is None):
                attrs['kernel_shape'] = (input_data2.shape[2], input_data2.shape[3], input_data2.shape[4])
            if (attrs.get('strides') is None):
                attrs['strides'] = (1, 1, 1)
            if (attrs.get('pads') is None):
                auto_pad = attrs.get('auto_pad')
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + attrs['kernel_shape'][2] - input_data1.shape[4])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + attrs['kernel_shape'][2] - input_data1.shape[4])/2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + attrs['kernel_shape'][2] - input_data1.shape[4])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + attrs['kernel_shape'][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + attrs['kernel_shape'][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + attrs['kernel_shape'][2] - input_data1.shape[4])/2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0,0,0,0,0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0,0,0,0,0,0)
#                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]+attrs['pads'][3])/attrs['strides'][0])+1,
                math.floor((input_data1.shape[3]-1-(attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+attrs['pads'][1]+attrs['pads'][4])/attrs['strides'][1])+1,
                math.floor((input_data1.shape[4]-1-(attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+attrs['pads'][2]+attrs['pads'][5])/attrs['strides'][2])+1)

            if cls.SIMD:
                work_pad_i = (node.input_tensor[0].shape[0] *
                              math.ceil(node.input_tensor[0].shape[1]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                            (node.input_tensor[0].shape[2] + attrs['pads'][0] + attrs['pads'][3]) * 
                            (node.input_tensor[0].shape[3] + attrs['pads'][1] + attrs['pads'][4]) * 
                            (node.input_tensor[0].shape[4] + attrs['pads'][2] + attrs['pads'][5]))
#                work_pad_w = np.prod(node.input_tensor[1].shape)                
                work_pad_w = (node.input_tensor[1].shape[0] *
                              node.input_tensor[1].shape[1] *
                              math.ceil(node.input_tensor[1].shape[2]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                              math.ceil(node.input_tensor[1].shape[3]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                              math.ceil(node.input_tensor[1].shape[4]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE)
#                work_pad_o = np.prod(outputs_shape)
                work_pad_o = (input_data1.shape[0] *
                             math.ceil(input_data2.shape[0]/cls.PARALLEL_SIZE)*cls.PARALLEL_SIZE *
                             (math.floor((input_data1.shape[2]-1-(attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+attrs['pads'][0]*2+attrs['pads'][3]*2)/attrs['strides'][0])+1) *
                             (math.floor((input_data1.shape[3]-1-(attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+attrs['pads'][1]*2+attrs['pads'][4]*2)/attrs['strides'][1])+1) *
                             (math.floor((input_data1.shape[4]-1-(attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+attrs['pads'][2]*2+attrs['pads'][5]*2)/attrs['strides'][2])+1))

                cls.WORK_PAD_I_SIZE = max(work_pad_i, cls.WORK_PAD_I_SIZE)
                cls.WORK_PAD_W_SIZE = max(work_pad_w, cls.WORK_PAD_W_SIZE)
                cls.WORK_PAD_O_SIZE = max(work_pad_o, cls.WORK_PAD_O_SIZE)
        
        else:
            raise(ValueError)
        if cls.SIMD:
            if (outputs_shape[1] < attrs['group'] * cls.PARALLEL_SIZE):
                cls.SIMD = False
#        if cls.SIMD:
#            if (outputs_shape[1] < attrs['group'] * cls.PARALLEL_SIZE):
#                cls.PARALLEL_SIZE = outputs_shape[1]//attrs['group']
        outputs_dtype = input_data1.dtype if input_data1.dtype == input_data2.dtype else np.double
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.empty(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=node.attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return 'ConvOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['conv.c']


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
                int   dilations[4];
                int   group;
                int   kernel_shape[2];
                int   pads[4];
                int   strides[2];
            }} ConvOpParam;
            '''
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)


    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res = ''
        if (cls.SIMD):
            res += '#define SIMD_VECTOR_SIZE ({})'.format(str(cls.PARALLEL_SIZE)) + '\n'
            res += '\n'
            res += '#ifndef ALIGN_SIZE' + '\n'
            res += '#define ALIGN_SIZE (sizeof(float) * SIMD_VECTOR_SIZE)' + '\n'
            res += '#endif' + '\n'
            res += '\n'
            res += '__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_i[{}];'.format(str(cls.WORK_PAD_I_SIZE)) + '\n'
            res += '__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_w[{}];'.format(str(cls.WORK_PAD_W_SIZE)) + '\n'
            res += '__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_o[{}];'.format(str(cls.WORK_PAD_O_SIZE)) + '\n'
            res += '\n'

        res += '#define mat_idx4(a, a_max, b, b_max, c, c_max, d, d_max) ((a)*(b_max)*(c_max)*(d_max) +(b)*(c_max)*(d_max) +(c)*(d_max) +(d))' + '\n'
        res += '#define mat_idx5(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max) +(b)*(c_max)*(d_max)*(e_max) +(c)*(d_max)*(e_max) +(d)*(e_max) +(e))' + '\n'
        res += '#define mat_idx6(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max, f, f_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max)*(f_max) +(b)*(c_max)*(d_max)*(e_max)*(f_max) +(c)*(d_max)*(e_max)*(f_max) +(d)*(e_max)*(f_max) +(e)*(f_max) +(f))' + '\n'
    
        return res


    def generate_c_code(self, **kwargs):
        res =''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +='\n\n'

        # param type
        res += self.get_c_param_type()
        res +='\n\n'

        # device
        use_simd = 0
        simd_vector_size = 1
        res += self.get_op_variale_def()
        res += '\n\n'

        ndim = self.output_tensor_ndims[0]
        if (ndim != 3 and ndim != 4 and ndim != 5):
            raise ValueError()

        dilations = self.attrs['dilations']
        group = self.attrs['group']
        kernel_shape = self.attrs['kernel_shape']
        pads = self.attrs['pads']
        strides = self.attrs['strides']

# define parameters (on python)
        if (ndim == 3):
            X_dn = self.input_tensor_shapes[0][0]
            X_dc = self.input_tensor_shapes[0][1]
            X_dd = 1
            X_dh = 1
            X_dw = self.input_tensor_shapes[0][2]
            W_dm = self.input_tensor_shapes[1][0]
            W_dc = self.input_tensor_shapes[1][1]
            W_dkd = 1
            W_dkh = 1
            W_dkw = self.input_tensor_shapes[1][2]
            Y_dn = self.output_tensor_shapes[0][0]
            Y_dc = self.output_tensor_shapes[0][1]
            Y_dd = 1
            Y_dh = 1
            Y_dw = self.output_tensor_shapes[0][2]
            aligned_ic = X_dc
            aligned_oc = Y_dc

            if (len(self.input_tensor) == 3):
                B_d0 = self.input_tensor_shapes[2][0]
            else:
                B_d0 = 0

            dilation_d = 1
            dilation_h = 1
            dilation_w = dilations[0]
            kernel_shape_d = 1
            kernel_shape_h = 1
            kernel_shape_w = kernel_shape[0] if (kernel_shape[0]!=0) else W_dkw
            pad_d_begin = 0
            pad_d_end = 0
            pad_h_begin = 0
            pad_h_end =   0
            pad_w_begin = pads[0]
            pad_w_end =   pads[1]
            padded_size_d = X_dd+pad_d_begin+pad_d_end
            padded_size_h = X_dh+pad_h_begin+pad_h_end
            padded_size_w = X_dw+pad_w_begin+pad_w_end
            stride_d = 1
            stride_h = 1
            stride_w = strides[0]
            unroll3x3 = 0
        elif (ndim == 4):
            X_dn = self.input_tensor_shapes[0][0]
            X_dc = self.input_tensor_shapes[0][1]
            X_dd = 1
            X_dh = self.input_tensor_shapes[0][2]
            X_dw = self.input_tensor_shapes[0][3]
            W_dm = self.input_tensor_shapes[1][0]
            W_dc = self.input_tensor_shapes[1][1]
            W_dkd = 1
            W_dkh = self.input_tensor_shapes[1][2]
            W_dkw = self.input_tensor_shapes[1][3]
            Y_dn = self.output_tensor_shapes[0][0]
            Y_dc = self.output_tensor_shapes[0][1]
            Y_dd = 1
            Y_dh = self.output_tensor_shapes[0][2]
            Y_dw = self.output_tensor_shapes[0][3]
            aligned_ic = X_dc
            aligned_oc = Y_dc

            if (len(self.input_tensor) == 3):  # with Bias
                B_d0 = self.input_tensor_shapes[2][0]
            else:
                B_d0 = 0

            dilation_d = 1
            dilation_h = dilations[0]
            dilation_w = dilations[1]
            kernel_shape_d = 1
            kernel_shape_h = kernel_shape[0] if (kernel_shape[0]!=0) else W_dkh
            kernel_shape_w = kernel_shape[1] if (kernel_shape[1]!=0) else W_dkw
            pad_d_begin = 0
            pad_d_end = 0
            pad_h_begin = pads[0]
            pad_h_end =   pads[2]
            pad_w_begin = pads[1]
            pad_w_end =   pads[3]
            padded_size_d = X_dd+pad_d_begin+pad_d_end
            padded_size_h = X_dh+pad_h_begin+pad_h_end
            padded_size_w = X_dw+pad_w_begin+pad_w_end
            stride_d = 1
            stride_h = strides[0]
            stride_w = strides[1]
            unroll3x3 = 0
        elif (ndim == 5):
            X_dn = self.input_tensor_shapes[0][0]
            X_dc = self.input_tensor_shapes[0][1]
            X_dd = self.input_tensor_shapes[0][2]
            X_dh = self.input_tensor_shapes[0][3]
            X_dw = self.input_tensor_shapes[0][4]
            W_dm = self.input_tensor_shapes[1][0]
            W_dc = self.input_tensor_shapes[1][1]
            W_dkd = self.input_tensor_shapes[1][2]
            W_dkh = self.input_tensor_shapes[1][3]
            W_dkw = self.input_tensor_shapes[1][4]
            Y_dn = self.output_tensor_shapes[0][0]
            Y_dc = self.output_tensor_shapes[0][1]
            Y_dd = self.output_tensor_shapes[0][2]
            Y_dh = self.output_tensor_shapes[0][3]
            Y_dw = self.output_tensor_shapes[0][4]
            aligned_ic = X_dc
            aligned_oc = Y_dc

            if (len(self.input_tensor) == 3):
                B_d0 = self.input_tensor_shapes[2][0]
            else:
                B_d0 = 0

            dilation_d = dilations[0]
            dilation_h = dilations[1]
            dilation_w = dilations[2]
            kernel_shape_d = kernel_shape[0] if (kernel_shape[0]!=0) else W_dkd
            kernel_shape_h = kernel_shape[1] if (kernel_shape[1]!=0) else W_dkh
            kernel_shape_w = kernel_shape[2] if (kernel_shape[2]!=0) else W_dkw
            pad_d_begin = pads[0]
            pad_d_end =   pads[3]
            pad_h_begin = pads[1]
            pad_h_end =   pads[4]
            pad_w_begin = pads[2]
            pad_w_end =   pads[5]
            padded_size_d = X_dd+pad_d_begin+pad_d_end
            padded_size_h = X_dh+pad_h_begin+pad_h_end
            padded_size_w = X_dw+pad_w_begin+pad_w_end
            stride_d = strides[0]
            stride_h = strides[1]
            stride_w = strides[2]
            unroll3x3 = 0

# define parameters (on C)
        if (ndim == 3):
            TemplateStatements = '''
            {t}* _X_pt = &X[0][0][0];
            {t}* _W_pt = &W[0][0][0];
            {t}* _Y_pt = &Y[0][0][0];
            '''
        elif (ndim == 4):
            TemplateStatements = '''
            {t}* _X_pt = &X[0][0][0][0];
            {t}* _W_pt = &W[0][0][0][0];
            {t}* _Y_pt = &Y[0][0][0][0];
            '''
        elif (ndim == 5):
            TemplateStatements = '''
            {t}* _X_pt = &X[0][0][0][0][0];
            {t}* _W_pt = &W[0][0][0][0][0];
            {t}* _Y_pt = &Y[0][0][0][0][0];
            '''

        TemplateStatements += '''
            const int  X_n = {X_dn};
            const int  X_c = {X_dc};
            const int  X_d = {X_dd};
            const int  X_h = {X_dh};
            const int  X_w = {X_dw};
            const int  aligned_X_c = {aligned_ic};
            const int  padded_X_d = {X_dd}+{pad_d_begin}+{pad_d_end};
            const int  padded_X_h = {X_dh}+{pad_h_begin}+{pad_h_end};
            const int  padded_X_w = {X_dw}+{pad_w_begin}+{pad_w_end};
            const int  W_m = {W_dm};
            const int  W_c = {W_dc};
            const int  W_kd = {W_dkd};
            const int  W_kh = {W_dkh};
            const int  W_kw = {W_dkw};
            const int  Y_n = {Y_dn};
            const int  Y_c = {Y_dc};
            const int  Y_d = {Y_dd};
            const int  Y_h = {Y_dh};
            const int  Y_w = {Y_dw};
            const int  aligned_Y_c = {aligned_oc};
            const int  padded_Y_d = {Y_dd}+{pad_d_begin}+{pad_d_end};
            const int  padded_Y_h = {Y_dh}+{pad_h_begin}+{pad_h_end};
            const int  padded_Y_w = {Y_dw}+{pad_w_begin}+{pad_w_end};
            const int  B_n = {B_d0};
            const int  dilation_d = {dilation_d};
            const int  dilation_h = {dilation_h};
            const int  dilation_w = {dilation_w};
            const int  group = {group};
            const int  kernel_shape_d = {kernel_shape_d};
            const int  kernel_shape_h = {kernel_shape_h};
            const int  kernel_shape_w = {kernel_shape_w};
            const int  pad_d_begin = {pad_d_begin};
            const int  pad_h_begin = {pad_h_begin};
            const int  pad_w_begin = {pad_w_begin};
            const int  pad_d_end = {pad_d_end};
            const int  pad_h_end = {pad_h_end};
            const int  pad_w_end = {pad_w_end};
            const int  stride_d = {stride_d};
            const int  stride_h = {stride_h};
            const int  stride_w = {stride_w};

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = {kernel_shape_d};
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = {kernel_shape_h};
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = {kernel_shape_w};
        '''

        if (ndim == 3) or (ndim == 4) or (ndim == 5):

            if (self.SIMD and (kernel_shape_h == 3) and (kernel_shape_w == 3) and (dilation_h == 1) and (dilation_w == 1) and ((Y_dc % self.PARALLEL_SIZE) == 0)):
                use_simd = 1
                unroll3x3 = 1
            if (self.SIMD and (kernel_shape_h == 1) and (kernel_shape_w == 1) and (dilation_h == 1) and (dilation_w == 1) and ((Y_dc % self.PARALLEL_SIZE) == 0)):
                use_simd = 1
                unroll3x3 = 0
            if (use_simd == 1):
                simd_vector_size = self.PARALLEL_SIZE
                aligned_ic = math.ceil(X_dc/simd_vector_size)*simd_vector_size
                aligned_oc = math.ceil(Y_dc/simd_vector_size)*simd_vector_size

                TemplateStatements += '''
// pre-process transpose
// padding & transpose input data : start
#ifdef TRANSPOSE
memset( (void *)work_pad_i, 0, sizeof(*_Y_pt) * X_n * aligned_X_c * (padded_X_d) * (padded_X_h) * (padded_X_w) );
#if TRANSPOSE == 1
                    for (n=0; n<X_n; n++) {{
#pragma omp parallel for
                        for (d=0; d<X_d; d++) {{
                            for (h=0; h<X_h; h++) {{
                                for (w=0; w<X_w; w++) {{
                                    for (ic=0; ic<X_c; ic++) {{
                                        work_pad_i[mat_idx5(n, X_n, (d+pad_d_begin), (padded_X_d), (h+pad_h_begin), (padded_X_h), (w+pad_w_begin), (padded_X_w), ic, aligned_X_c)] = *(_X_pt + mat_idx5(n, X_n, ic, X_c, d, X_d, h, X_h, w, X_w));
                                    }}
                                }}
                            }}
                        }}
                    }}
#elif TRANSPOSE == 2
                '''
                if (Y_dc == X_dc) and (X_dc == group): # group != 1 and oc == ic == group
                    TemplateStatements += '''
                    for (n=0; n<X_n; n++) {{
#pragma omp parallel for
                        for (oc1=0; oc1<X_c/SIMD_VECTOR_SIZE; oc1++) {{
                            for (d=0; d<X_d; d++) {{
                                for (h=0; h<X_h; h++) {{
                                    for (w=0; w<X_w; w++) {{
                                        for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                            work_pad_i[mat_idx6(n, X_n, oc1, aligned_X_c/SIMD_VECTOR_SIZE, (d+pad_d_begin), (padded_X_d), (h+pad_h_begin), (padded_X_h), (w+pad_w_begin), (padded_X_w), oc2, SIMD_VECTOR_SIZE)] = *(_X_pt + mat_idx5(n, X_n, (oc1*SIMD_VECTOR_SIZE+oc2), X_c, d, X_d, h, X_h, w, X_w));
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                    '''
                else:
                    TemplateStatements += '''
                    for (n=0; n<X_n; n++) {{
#pragma omp parallel for
                        for (ic=0; ic<X_c; ic++) {{
                            for (d=0; d<X_d; d++) {{
                                for (h=0; h<X_h; h++) {{
                                    for (w=0; w<X_w; w++) {{
                                        work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), (padded_X_d), (h+pad_h_begin), (padded_X_h), (w+pad_w_begin), (padded_X_w))] = *(_X_pt + mat_idx5(n, X_n, ic, X_c, d, X_d, h, X_h, w, X_w));
                                    }}
                                }}
                            }}
                        }}
                    }}
                    '''
#                    TemplateStatements += '''
#                    for (n=0; n<X_n; n++) {{
##pragma omp parallel for
#                        for (d=0; d<X_d; d++) {{
#                            for (h=0; h<X_h; h++) {{
#                                for (w=0; w<X_w; w++) {{
#                                    for (ic=0; ic<X_c; ic++) {{
#                                        work_pad_i[mat_idx5(n, X_n, (d+pad_d_begin), (padded_X_d), (h+pad_h_begin), (padded_X_h), (w+pad_w_begin), (padded_X_w), ic, aligned_X_c)] = *(_X_pt + mat_idx5(n, X_n, ic, X_c, d, X_d, h, X_h, w, X_w));
#                                    }}
#                                }}
#                            }}
#                        }}
#                    }}
#                    '''
                TemplateStatements += '''
#endif  // TRANSPOSE type
// padding & transpose input data : end
#endif  // TRANSPOSE
                '''

                TemplateStatements += '''
// transpose weight : start
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (kd=0; kd<kernel_shape_d; kd++) {{
                    for (kh=0; kh<kernel_shape_h; kh++) {{
                        for (kw=0; kw<kernel_shape_w; kw++) {{
                            for (ic=0; ic<aligned_X_c/group; ic++) {{
                                for (oc=0; oc<aligned_Y_c; oc++) {{
                                    work_pad_w[mat_idx5(kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w, ic, aligned_X_c/group, oc, aligned_Y_c)] = *(_W_pt + mat_idx5( oc, Y_c, ic/group, X_c/group, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w ));
                                }}
                            }}
                        }}
                    }}
                }}

#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<aligned_Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (ic=0; ic<X_c/group; ic++) {{
                        for (kd=0; kd<kernel_shape_d; kd++) {{
                            for (kh=0; kh<kernel_shape_h; kh++) {{
                                for (kw=0; kw<kernel_shape_w; kw++) {{
                                    for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                        work_pad_w[mat_idx6(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c/group, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)] = *(_W_pt + mat_idx5( (oc1*SIMD_VECTOR_SIZE+oc2), Y_c, ic/group, X_c/group, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE
// padding & transpose weight : end
                '''

                TemplateStatements += '''
// bias settings
#ifdef TRANSPOSE
                '''
                if (B_d0 == 0):
                    TemplateStatements += '''
            memset( (void *)work_pad_o, 0, sizeof(*_Y_pt) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    '''
                TemplateStatements += '''
            for (n=0; n<Y_n; n++) {{
                '''
                if (B_d0 != 0):
                    TemplateStatements += '''
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (d=0; d<Y_d; d++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            for (oc=0; oc<aligned_Y_c; oc++) {{
                                work_pad_o[mat_idx5(n, Y_n, d, Y_d, h, Y_h, w, Y_w, oc, aligned_Y_c)] = B[oc];
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<aligned_Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (d=0; d<Y_d; d++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                    work_pad_o[mat_idx6(n, Y_n, oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] = B[oc1*SIMD_VECTOR_SIZE+oc2];
                                }}
                            }}
                        }}
                    }}
                }}
#endif // TRANSPOSE TYPE
                    '''

                if (group == 1):
                    TemplateStatements += '''
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (d=0; d<Y_d; d++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            for (ic=0; ic<X_c; ic++) {{
                                for (oc=0; oc<Y_c; oc++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
                                    work_pad_o[mat_idx5(n, Y_n, d, Y_d, h, Y_h, w, Y_w, oc, aligned_Y_c)] += work_pad_i[mat_idx5(n, X_n, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), padded_X_h, (w*stride_w+{kw}), padded_X_w, ic, aligned_X_c)] * work_pad_w[mat_idx5({kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, ic, aligned_X_c, oc, Y_c)];
                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                                }}
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
                                        work_pad_o[mat_idx6(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), (padded_X_h), (w*stride_w+{kw}), (padded_X_w))] * work_pad_w[mat_idx6(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, {kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                '''.format(**unroll_mapping)
#                                TemplateStatements += '''
#                                        work_pad_o[mat_idx6(n, Y_n, oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx5(n, X_n, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), (padded_X_h), (w*stride_w+{kw}), (padded_X_w), ic, aligned_X_c)] * work_pad_w[mat_idx6(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, {kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
#                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE TYPE
            }}
                    '''
                elif (Y_dc == X_dc) and (X_dc == group): # group != 1 and oc == ic == group
                    TemplateStatements += '''
// convolution process
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (d=0; d<Y_d; d++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            for (oc=0; oc<Y_c; oc++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
                                    work_pad_o[mat_idx5(n, Y_n, d, Y_d, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx5(n, X_n, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), padded_X_h, (w*stride_w+{kw}), padded_X_w, oc, aligned_X_c)] * work_pad_w[mat_idx5({kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc/group, aligned_X_c/group, oc, Y_c)];
                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (d=0; d<Y_d; d++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
//                                    work_pad_o[mat_idx6(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx5(n, X_n, (oc1*SIMD_VECTOR_SIZE+oc2), aligned_X_c, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), (padded_X_h), (w*stride_w+{kw}), (padded_X_w))] * work_pad_w[mat_idx6(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), (oc1*SIMD_VECTOR_SIZE+oc2)/group, aligned_X_c/group, {kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx6(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx6(n, X_n, oc1, aligned_X_c/SIMD_VECTOR_SIZE, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), (padded_X_h), (w*stride_w+{kw}), (padded_X_w), oc2, SIMD_VECTOR_SIZE)] * work_pad_w[mat_idx6(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), 0, 1, {kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE TYPE
            }}
                    '''
                else: # group != 1 and oc != ic
                    TemplateStatements += '''
// convolution process
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (d=0; d<Y_d; d++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            for (ic=0; ic<X_c/group; ic++) {{
                                for (oc=0; oc<Y_c/group; oc++) {{
                                    for (int current_group=0; current_group<group; current_group++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
                                        work_pad_o[mat_idx5(n, Y_n, d, Y_d, h, Y_h, w, Y_w, current_group*Y_c/group+oc, Y_c)] += work_pad_i[mat_idx5(n, X_n, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), padded_X_h, (w*stride_w+{kw}), padded_X_w, current_group*aligned_X_c/group+ic, aligned_X_c)] * work_pad_w[mat_idx5({kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, ic, aligned_X_c/group, current_group*Y_c/group+oc, Y_c)];
                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (int current_group=0; current_group<group; current_group++) {{
                    for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE/group; oc1++) {{
                        for (ic=0; ic<X_c/group; ic++) {{
                            for (d=0; d<Y_d; d++) {{
                                for (h=0; h<Y_h; h++) {{
                                    for (w=0; w<Y_w; w++) {{
                                        for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                    '''
                    for kd in list(range(kernel_shape_d)):
                        for kh in list(range(kernel_shape_h)):
                            for kw in list(range(kernel_shape_w)):
                                unroll_mapping = {}
                                unroll_mapping.update({'kd': kd})
                                unroll_mapping.update({'kh': kh})
                                unroll_mapping.update({'kw': kw})
                                TemplateStatements += '''
                                            work_pad_o[mat_idx6(n, Y_n, current_group*Y_c/group/SIMD_VECTOR_SIZE+oc1, (Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx5(n, X_n, current_group*aligned_X_c/group+ic, aligned_X_c/group, (d*stride_d+{kd}), padded_X_d, (h*stride_h+{kh}), (padded_X_h), (w*stride_w+{kw}), (padded_X_w))] * work_pad_w[mat_idx6(current_group*aligned_Y_c/SIMD_VECTOR_SIZE/group+oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c/group, {kd}, kernel_shape_d, {kh}, kernel_shape_h, {kw}, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                '''.format(**unroll_mapping)
                    TemplateStatements += '''
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE TYPE
            }}
                    '''

                TemplateStatements += '''
// post-process transpose start
#if TRANSPOSE == 1  // M_MINOR
                for (n=0; n<Y_n; n++) {{
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = work_pad_o[mat_idx5(n, Y_n, d, Y_d, h, Y_h, w, Y_w, oc, aligned_Y_c)];
                                }}
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
                for (n=0; n<Y_n; n++) {{
#pragma omp parallel for
                    for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                        for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                            for (d=0; d<Y_d; d++) {{
                                for (h=0; h<Y_h; h++) {{
                                    for (w=0; w<Y_w; w++) {{
                                        *(_Y_pt + mat_idx5(n, Y_n, (oc1*SIMD_VECTOR_SIZE+oc2), Y_c, d, Y_d, h, Y_h, w, Y_w)) = work_pad_o[mat_idx6(n, Y_n, oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), d, Y_d, h, Y_h, w, Y_w, oc2, SIMD_VECTOR_SIZE)];
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE TYPE
#endif  // TRANSPOSE
// transpose end

                '''

                TemplateStatements += '''
// pre-process padding
#ifndef TRANSPOSE
// no transpose, padding only input data
                for (n=0; n<X_n; n++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (d=0; d<pad_d_begin; d++) {{
                            for (h=0; h<padded_X_h; h++) {{
                                for (w=0; w<padded_X_w; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, d, padded_X_d, h, padded_X_h, w, padded_X_w)] = 0.0f;
                                }}
                            }}
                        }}
                        for (d=0; d<X_d; d++) {{
                            for (h=0; h<pad_h_begin; h++) {{
                                for (w=0; w<padded_X_w; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), padded_X_d, h, padded_X_h, w, padded_X_w)] = 0.0f;
                                }}
                            }}
                            for (h=0; h<X_h; h++) {{
                                for (w=0; w<pad_w_begin; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), padded_X_d, (h+pad_h_begin), padded_X_h, w, padded_X_w)] = 0.0f;
                                }}
                                for (w=0; w<X_w; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), padded_X_d, (h+pad_h_begin), padded_X_h, (w+pad_w_begin), padded_X_w)] = *(_X_pt + mat_idx5(n, X_n, ic, X_c, d, X_d, h, X_h, w, X_w));
                                }}
                                for (w=0; w<pad_w_end; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), padded_X_d, (h+pad_h_begin), padded_X_h, (w+pad_w_begin+X_w), padded_X_w)] = 0.0f;
                                }}
                            }}
                            for (h=0; h<pad_h_end; h++) {{
                                for (w=0; w<padded_X_w; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin), padded_X_d, (h+pad_h_begin+X_h), padded_X_h, w, padded_X_w)] = 0.0f;
                                }}
                            }}
                        }}
                        for (d=0; d<pad_d_end; d++) {{
                            for (h=0; h<padded_X_h; h++) {{
                                for (w=0; w<padded_X_w; w++) {{
                                    work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, (d+pad_d_begin+X_d), padded_X_d, h, padded_X_h, w, padded_X_w)] = 0.0f;
                                }}
                            }}
                        }}
                    }}
                }}
                '''

                if (group == 1):
                    if (B_d0 == 0):
                        TemplateStatements += '''
// Bias settings
                memset( (void *)Y, 0, sizeof(*_Y_pt) * Y_n * Y_c * Y_d * Y_h * Y_w );
                        '''
                    TemplateStatements += '''
                for (n=0; n<Y_n; n++) {{
                    '''
                    if (B_d0 != 0):
                        TemplateStatements += '''
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                }}
                            }}
                        }}
                    }}
                        '''
                    TemplateStatements += '''
// convolution process
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (ic=0; ic<X_c; ic++) {{
                            for (d=0; d<Y_d; d++) {{
                                for (h=0; h<Y_h; h++) {{
                                    for (w=0; w<Y_w; w++) {{
                                        for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                            current_d = d*stride_d+kd*dilation_d;
                                            for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                                current_h = h*stride_h+kh*dilation_h;
                                                for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                    current_w = w*stride_w+kw*dilation_w;
                                                    *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += work_pad_i[mat_idx5(n, X_n, ic, aligned_X_c, current_d, padded_X_d, current_h, padded_X_h, current_w, padded_X_w)] * *(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif   // !TRANSPOSE
                    '''
                elif (Y_dc == X_dc) and (X_dc == group): # group != 1 and oc == ic == group
                    if (B_d0 == 0):
                        TemplateStatements += '''
// Bias settings
                memset( (void *)Y, 0, sizeof(*_Y_pt) * Y_n * Y_c * Y_d * Y_h * Y_w );
                        '''
                    TemplateStatements += '''
                for (n=0; n<Y_n; n++) {{
                    '''
                    if (B_d0 != 0):
                        TemplateStatements += '''
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                }}
                            }}
                        }}
                    }}
                        '''
                    TemplateStatements += '''
// convolution process
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                        current_d = d*stride_d+kd*dilation_d;
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            current_h = h*stride_h+kh*dilation_h;
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                current_w = w*stride_w+kw*dilation_w;
                                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += work_pad_i[mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)] * *(_W_pt + mat_idx5(oc, Y_c, oc/group, X_c/group, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif   // !TRANSPOSE
                    '''
                else: # group != 1 and oc != ic
                    if (B_d0 == 0):
                        TemplateStatements += '''
// Bias settings
                memset( (void *)Y, 0, sizeof(*_Y_pt) * Y_n * Y_c * Y_d * Y_h * Y_w );
                        '''
                    TemplateStatements += '''
                for (n=0; n<Y_n; n++) {{
                    '''
                    if (B_d0 != 0):
                        TemplateStatements += '''
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                }}
                            }}
                        }}
                    }}
                        '''
                    TemplateStatements += '''
// convolution process
#pragma omp parallel for
                    for (int current_group=0; current_group<group; current_group++) {{
                        for (oc=0; oc<Y_c/group; oc++) {{
                            for (ic=0; ic<X_c/group; ic++) {{
                                for (d=0; d<Y_d; d++) {{
                                    for (h=0; h<Y_h; h++) {{
                                        for (w=0; w<Y_w; w++) {{
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                                current_d = d*stride_d+kd*dilation_d;
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                                    current_h = h*stride_h+kh*dilation_h;
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                        current_w = w*stride_w+kw*dilation_w;
                                                        *(_Y_pt + mat_idx5(n, Y_n, current_group*group+oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += work_pad_i[mat_idx5(n, X_n, current_group*X_c/group+ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)] * *(_W_pt + mat_idx5(current_group*Y_c/group+oc, Y_c, ic, X_c/group, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                                    }}
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
#endif   // !TRANSPOSE
                    '''

            else: # CPU
                if (B_d0 == 0):
                    TemplateStatements += '''
            memset( (void *)Y, 0, sizeof(*_Y_pt) * Y_n * Y_c * Y_h * Y_w );
                    '''

                TemplateStatements += '''
            for (n=0; n<Y_n; n++) {{
                '''

                if (B_d0 != 0):
                    TemplateStatements += '''
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {{
                    for (d=0; d<Y_d; d++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                            }}
                        }}
                    }}
                }}
                '''
                if (group == 1):
                    TemplateStatements += '''
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) {{ continue; }}
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) {{ continue; }}
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) {{ continue; }}
                                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += *(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w))
                                                                                                                *  *(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                    '''
                elif (Y_dc == X_dc) and (X_dc == group): # group != 1 and oc == ic == group
                    TemplateStatements += '''
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) {{ continue; }}
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) {{ continue; }}
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) {{ continue; }}
                                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += *(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w))
                                                                                                              * *(_W_pt + mat_idx5(oc, Y_c, (oc/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                    '''
                else: # group != 1 and oc != ic
                    TemplateStatements += '''
#pragma omp parallel for
                for (int current_group=0; current_group<group; current_group++) {{
                    for (oc=0; oc<Y_c/group; oc++) {{
                        for (ic=0; ic<X_c/group; ic++) {{
                            for (d=0; d<Y_d; d++) {{
                                for (h=0; h<Y_h; h++) {{
                                    for (w=0; w<Y_w; w++) {{
                                        for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                            current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                            if (current_d<0 || current_d>=X_d) {{ continue; }}
                                            for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                                current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                if (current_h<0 || current_h>=X_h) {{ continue; }}
                                                for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                    current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                    if (current_w<0 || current_w>=X_w) {{ continue; }}
                                                    *(_Y_pt + mat_idx5(n, Y_n, current_group*Y_c/group+oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) += *(_X_pt + mat_idx5(n, X_n, current_group*X_c/group+ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w))
                                                                                                                * *(_W_pt + mat_idx5(current_group*Y_c/group+oc, Y_c, ic, (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w));
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
                    '''

                TemplateStatements += '''
            }}
                '''

# common mapping
        mapping = {}
        mapping.update({'X_dn': X_dn})
        mapping.update({'X_dc': X_dc})
        mapping.update({'X_dd': X_dd})
        mapping.update({'X_dh': X_dh})
        mapping.update({'X_dw': X_dw})
        mapping.update({'W_dm': W_dm})
        mapping.update({'W_dc': W_dc})
        mapping.update({'W_dkd': W_dkd})
        mapping.update({'W_dkh': W_dkh})
        mapping.update({'W_dkw': W_dkw})
        mapping.update({'Y_dn': Y_dn})
        mapping.update({'Y_dc': Y_dc})
        mapping.update({'Y_dd': Y_dd})
        mapping.update({'Y_dh': Y_dh})
        mapping.update({'Y_dw': Y_dw})
        mapping.update({'aligned_ic': aligned_ic})
        mapping.update({'aligned_oc': aligned_oc})
        mapping.update({'B_d0': B_d0})
        mapping.update({'dilation_d': dilation_d})
        mapping.update({'dilation_h': dilation_h})
        mapping.update({'dilation_w': dilation_w})
        mapping.update({'group': group})
        mapping.update({'kernel_shape_d': kernel_shape_d})
        mapping.update({'kernel_shape_h': kernel_shape_h})
        mapping.update({'kernel_shape_w': kernel_shape_w})
        mapping.update({'pad_d_begin': pad_d_begin})
        mapping.update({'pad_d_end':   pad_d_end})
        mapping.update({'pad_h_begin': pad_h_begin})
        mapping.update({'pad_h_end':   pad_h_end})
        mapping.update({'pad_w_begin': pad_w_begin})
        mapping.update({'pad_w_end':   pad_w_end})
        mapping.update({'padded_size_d': padded_size_d})
        mapping.update({'padded_size_h': padded_size_h})
        mapping.update({'padded_size_w': padded_size_w})
        mapping.update({'stride_d': stride_d})
        mapping.update({'stride_h': stride_h})
        mapping.update({'stride_w': stride_w})
        mapping.update({'unroll3x3': unroll3x3})
        mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})


        # 3        
        if (use_simd == 1):
            TemplateFunction = '''
//#define TRANSPOSE 1  // M_MINOR
#define TRANSPOSE 2  // M_SEPARATE
//#define TRANSPOSE 3 // no-pad, just transpose
//#undef TRANSPOSE
            '''
        else:
            TemplateFunction = '''
#undef TRANSPOSE
            '''
        if (len(self.input_tensor) == 3):
            TemplateFunction += cleandoc('''
            void {op_func_name}(void *op_param, {t} X{dims_X}, {t} W{dims_W}, {t} B{dims_B}, {t} Y{dims}, void *inputs_params, void* outputs_params)
            {{
                {statements}
            }}
            ''')
        else:
            TemplateFunction += cleandoc('''
            void {op_func_name}(void *op_param, {t} X{dims_X}, {t} W{dims_W}, {t} Y{dims}, void *inputs_params, void* outputs_params)
            {{
                {statements}
            }}
            ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'dims_X': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({'W': self.input_tensor_names[1]})
        mappingf.update({'dims_W': c_helper.generate_dim_bracket(self.input_tensor_shapes[1])}) 
        if (len(self.input_tensor) == 3):
            mappingf.update({'B': self.input_tensor_names[2]})
            mappingf.update({'dims_B': c_helper.generate_dim_bracket(self.input_tensor_shapes[2])}) 
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
    def need_c_headers(cls):
        return ['string.h']

    
    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)






