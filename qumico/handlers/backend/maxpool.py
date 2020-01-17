from inspect import cleandoc
import math

import numpy as np

from onnx.backend.base import namedtupledict
from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op


@onnx_op('MaxPool')
class MaxPool(BackendHandler):

    OpenMP=False

#Maxpool-8 is equal to case of dilation[i]=1 and ceil_mode=0 in Maxpool-10

#output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

#Maxpool-10(default)
#output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
#Maxpool-10(ceil_mode)
#output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

#Maxpool-8
#VALID
# output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
#SAME
# output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
# pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]

#Maxpool-10
#VALID
# output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
#SAME
# output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
# pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]


    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]
        attrs = node.attrs

        if (attrs.get('ceil_mode') == None):       # define ceil_mode after Maxpool-10. default is 0.
            attrs['ceil_mode'] = 0

        if (input_data1.ndim == 3):
            if (attrs.get('strides') == None):
                attrs['strides'] = (1,)
            if (attrs.get('dilations') == None):    # define dilations[] after Maxpool-10.
                attrs['dilations'] = (1,)

            # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get('auto_pad')
            if (attrs.get('pads') == None):
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0)
                elif (auto_pad == 'NOTSET' or auto_pad == None):
                    attrs['pads'] = (0,0)
            if (attrs.get('storage_order') == None):
                attrs['storage_order'] = 0

            # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
            # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
            # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-1):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == 'SAME_UPPER') or (auto_pad == 'SAME_LOWER'):
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs['strides'][-1]))
            elif (auto_pad == 'VALID'):
                tmp_shape.append(math.ceil((input_data1.shape[-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1) +1)/attrs['strides'][-1]))
            else: # auto_pad is None
                if (attrs['ceil_mode'] == 0):
                    tmp_shape.append(math.floor((input_data1.shape[-1] + attrs['pads'][0] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))
                else:
                    tmp_shape.append(math.ceil((input_data1.shape[-1] + attrs['pads'][0] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))

        elif (input_data1.ndim == 4):
            if (attrs.get('strides') == None):
                attrs['strides'] = (1,1)
            if (attrs.get('dilations') == None):    # define dilations[] after Maxpool-10.
                attrs['dilations'] = (1,1)

            # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get('auto_pad')
            if (attrs.get('pads') == None):
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0,0,0)
                elif (auto_pad == 'NOTSET' or auto_pad == None):
                    attrs['pads'] = (0,0,0,0)
            if (attrs.get('storage_order') == None):
                attrs['storage_order'] = 0

            # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
            # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
            # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-2):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == 'SAME_UPPER') or (auto_pad == 'SAME_LOWER'):
                tmp_shape.append(math.ceil(input_data1.shape[-2]/attrs['strides'][-2]))
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs['strides'][-1]))
            elif (auto_pad == 'VALID'):
                tmp_shape.append(math.ceil((input_data1.shape[-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1) +1)/attrs['strides'][-2]))
                tmp_shape.append(math.ceil((input_data1.shape[-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1) +1)/attrs['strides'][-1]))
            else: # auto_pad is None
                if (attrs['ceil_mode'] == 0):
                    tmp_shape.append(math.floor((input_data1.shape[-2] + attrs['pads'][0] + attrs['pads'][-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1))/attrs['strides'][-2] +1))
                    tmp_shape.append(math.floor((input_data1.shape[-1] + attrs['pads'][1] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))
                else:
                    tmp_shape.append(math.ceil((input_data1.shape[-2] + attrs['pads'][0] + attrs['pads'][-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1))/attrs['strides'][-2] +1))
                    tmp_shape.append(math.ceil((input_data1.shape[-1] + attrs['pads'][1] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))

        elif (input_data1.ndim == 5):
            if (attrs.get('strides') == None):
                attrs['strides'] = (1,1,1)
            if (attrs.get('dilations') == None):    # define dilations[] after Maxpool-10.
                attrs['dilations'] = (1,1,1)

            # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get('auto_pad')
            if (attrs.get('pads') == None):
                if (auto_pad == 'SAME_UPPER'):
                    attrs['pads'] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + ((attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+1) - input_data1.shape[4])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + ((attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+1) - input_data1.shape[4])/2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + ((attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+1) - input_data1.shape[4])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs['strides'][0]) -1) * attrs['strides'][0] + ((attrs['kernel_shape'][0]-1)*attrs['dilations'][0]+1) - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs['strides'][1]) -1) * attrs['strides'][1] + ((attrs['kernel_shape'][1]-1)*attrs['dilations'][1]+1) - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs['strides'][2]) -1) * attrs['strides'][2] + ((attrs['kernel_shape'][2]-1)*attrs['dilations'][2]+1) - input_data1.shape[4])/2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0,0,0,0,0,0)
                elif (auto_pad == 'NOTSET' or auto_pad == None):
                    attrs['pads'] = (0,0,0,0,0,0)
            if (attrs.get('storage_order') == None):
                attrs['storage_order'] = 0

            # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
            # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
            # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-3):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == 'SAME_UPPER') or (auto_pad == 'SAME_LOWER'):
                tmp_shape.append(math.ceil(input_data1.shape[-3]/attrs['strides'][-3]))
                tmp_shape.append(math.ceil(input_data1.shape[-2]/attrs['strides'][-2]))
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs['strides'][-1]))
            elif (auto_pad == 'VALID'):
                tmp_shape.append(math.ceil((input_data1.shape[-3] - ((attrs['kernel_shape'][-3]-1)*attrs['dilations'][-3]+1) +1)/attrs['strides'][-3]))
                tmp_shape.append(math.ceil((input_data1.shape[-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1) +1)/attrs['strides'][-2]))
                tmp_shape.append(math.ceil((input_data1.shape[-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1) +1)/attrs['strides'][-1]))
            else: # auto_pad is None
                if (attrs['ceil_mode'] == 0):
                    tmp_shape.append(math.floor((input_data1.shape[-3] + attrs['pads'][0] + attrs['pads'][-3] - ((attrs['kernel_shape'][-3]-1)*attrs['dilations'][-3]+1))/attrs['strides'][-3] +1))
                    tmp_shape.append(math.floor((input_data1.shape[-2] + attrs['pads'][1] + attrs['pads'][-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1))/attrs['strides'][-2] +1))
                    tmp_shape.append(math.floor((input_data1.shape[-1] + attrs['pads'][2] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))
                else:
                    tmp_shape.append(math.ceil((input_data1.shape[-3] + attrs['pads'][0] + attrs['pads'][-3] - ((attrs['kernel_shape'][-3]-1)*attrs['dilations'][-3]+1))/attrs['strides'][-3] +1))
                    tmp_shape.append(math.ceil((input_data1.shape[-2] + attrs['pads'][1] + attrs['pads'][-2] - ((attrs['kernel_shape'][-2]-1)*attrs['dilations'][-2]+1))/attrs['strides'][-2] +1))
                    tmp_shape.append(math.ceil((input_data1.shape[-1] + attrs['pads'][2] + attrs['pads'][-1] - ((attrs['kernel_shape'][-1]-1)*attrs['dilations'][-1]+1))/attrs['strides'][-1] +1))
        else:
            raise(ValueError)

        outputs_shape = tuple(tmp_shape)
        outputs_dtype = input_data1.dtype
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict('output_tensor', outputs_dict.keys())(**outputs_dict)

        device = kwargs.get('device')
        if (issubclass(device.__class__, QumicoDevice) and 
            QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True
        
        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return 'MaxPoolOpParam'


    @classmethod
    def get_c_op_file_name(cls):
        return ['maxpool.c']


    @classmethod
    @BackendHandler.dec_generate_once(resType=list)
    def get_c_op_include_header(cls):
        return ['math.h', 'float.h']
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            '''
            typedef struct {{
                char* name;
                int   ceil_mode;
                int   dilations[3];
                int   kernel_shape[3];
                int   pads[6];
                int   storage_order;
                int   strides[3];
            }} MaxPoolOpParam;
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

        ndim = self.output_tensor_ndims[0]
        if (ndim != 3 and ndim != 4 and ndim != 5):
            raise ValueError()

        kernel_shape = self.attrs['kernel_shape']
        pads = self.attrs['pads']
        storage_order = self.attrs['storage_order']
        strides = self.attrs['strides']
        dilations = self.attrs['dilations']

        if (ndim == 3):
            TemplateStatements = '''
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_w = {X_d2};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_w = {Y_d2};
                const int  kernel_shape_w = {kernel_shape_w};
                const int  pad_w_begin = {pad_w_begin};
                const int  pad_w_end = {pad_w_end};
                const int  stride_w = {stride_w};
                const int  dilation_w = {dilation_w};
                const int  storage_order = {storage_order};

                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0]) * Y_n * Y_c * Y_w );

                for (int n=0; n<Y_n; n++) {{
                    {pragma}
                    for (int c=0; c<Y_c; c++) {{
                        for (int w=0; w<Y_w; w++) {{
                            {t} pool;
                            int  max_flag;
                            pool = -DBL_MAX;
                            max_flag = 0;
                            for (int kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                if ((w*stride_w+kw*dilation_w < 0) || (w*stride_w+kw*dilation_w >= X_w)) {{ continue; }}
                                if (pool < X[n][c][w*stride_w+kw*dilation_w]) {{
                                    pool = X[n][c][w*stride_w+kw*dilation_w];
                                    max_flag = 1;
                                }}
                            }}
                            if (max_flag) {{
                                Y[n][c][w] = pool;
                            }}
                        }}
                    }}
                }}
            '''
            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})
            mapping.update({'kernel_shape_w': kernel_shape[0]})
            mapping.update({'pad_w_begin': pads[0]})
            mapping.update({'pad_w_end':   pads[1]})
            mapping.update({'stride_w': strides[0]})
            mapping.update({'dilation_w': dilations[0]})
            mapping.update({'storage_order': storage_order})
            mapping.update({'pragma':self.PRAGMA_OMP if self.OpenMP else ''})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})

        elif (ndim == 4):
            TemplateStatements = '''
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_h = {X_d2};
                const int  X_w = {X_d3};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_h = {Y_d2};
                const int  Y_w = {Y_d3};
                const int  kernel_shape_h = {kernel_shape_h};
                const int  kernel_shape_w = {kernel_shape_w};
                const int  pad_h_begin = {pad_h_begin};
                const int  pad_w_begin = {pad_w_begin};
                const int  pad_h_end = {pad_h_end};
                const int  pad_w_end = {pad_w_end};
                const int  stride_h = {stride_h};
                const int  stride_w = {stride_w};
                const int  dilation_h = {dilation_h};
                const int  dilation_w = {dilation_w};
                const int  storage_order = {storage_order};

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );
            '''

            if ((pads[0] == 0) and (pads[1] == 0) and (pads[2] == 0) and (pads[3] == 0) and (dilations[0] == 1) and (dilations[1] == 1)
                            and (kernel_shape[0] == 2) and (kernel_shape[1] == 2)
                            and (self.input_tensor_shapes[0][2] % strides[0] == 0) and (self.input_tensor_shapes[0][3] % strides[1] == 0)):
                TemplateStatements += '''
                    for (int n=0; n<Y_n; n++) {{
                        {pragma}
                        for (int c=0; c<Y_c; c++) {{
                            if (storage_order == 0) {{
                                for (int h=0; h<Y_h; h++) {{
                                    for (int w=0; w<Y_w; w++) {{
                                        Y[n][c][h][w] = fmaxf(
                                            fmaxf( X[n][c][h*stride_h+0][w*stride_w+0], X[n][c][h*stride_h+0][w*stride_w+1] ),
                                            fmaxf( X[n][c][h*stride_h+1][w*stride_w+0], X[n][c][h*stride_h+1][w*stride_w+1] ));
                                    }}
                                }}
                            }} else {{
                                for (int w=0; w<Y_w; w++) {{
                                    for (int h=0; h<Y_h; h++) {{
                                        Y[n][c][h][w] = fmaxf(
                                            fmaxf( X[n][c][h*stride_h+0][w*stride_w+0], X[n][c][h*stride_h+0][w*stride_w+1] ),
                                            fmaxf( X[n][c][h*stride_h+1][w*stride_w+0], X[n][c][h*stride_h+1][w*stride_w+1] ));
                                    }}
                                }}
                            }}
                        }}
                    }}
                '''
            else:
                TemplateStatements += '''
                    for (int n=0; n<Y_n; n++) {{
                        {pragma}
                        for (int c=0; c<Y_c; c++) {{
                            if (storage_order == 0) {{
                                for (int h=0; h<Y_h; h++) {{
                                    for (int w=0; w<Y_w; w++) {{
                                        {t} pool;
                                        int  max_flag;
                                        pool = -DBL_MAX;
                                        max_flag = 0;
                                        for (int kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            if ((h*stride_h+kh*dilation_h < 0) || (h*stride_h+kh*dilation_h >= X_h)) {{ continue; }}
                                            for (int kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                if ((w*stride_w+kw*dilation_w < 0) || (w*stride_w+kw*dilation_w >= X_w)) {{ continue; }}
                                                if (pool < X[n][c][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w]) {{
                                                    pool = X[n][c][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w];
                                                    max_flag = 1;
                                                }}
                                            }}
                                        }}
                                        if (max_flag) {{
                                            Y[n][c][h][w] = pool;
                                        }}
                                    }}
                                }}
                            }} else {{
                                for (int w=0; w<Y_w; w++) {{
                                    for (int h=0; h<Y_h; h++) {{
                                        {t} pool;
                                        int  max_flag;
                                        pool = -DBL_MAX;
                                        max_flag = 0;
                                        for (int kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            if ((h*stride_h+kh*dilation_h < 0) || (h*stride_h+kh*dilation_h >= X_h)) {{ continue; }}
                                            for (int kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                if ((w*stride_w+kw*dilation_w < 0) || (w*stride_w+kw*dilation_w >= X_w)) {{ continue; }}
                                                if (pool < X[n][c][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w]) {{
                                                    pool = X[n][c][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w];
                                                    max_flag = 1;
                                                }}
                                            }}
                                        }}
                                        if (max_flag) {{
                                            Y[n][c][h][w] = pool;
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                '''

            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'X_d3': self.input_tensor_shapes[0][3]})
            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})
            mapping.update({'Y_d3': self.output_tensor_shapes[0][3]})
            mapping.update({'kernel_shape_h': kernel_shape[0]})
            mapping.update({'kernel_shape_w': kernel_shape[1]})
            mapping.update({'pad_h_begin': pads[0]})
            mapping.update({'pad_h_end':   pads[2]})
            mapping.update({'pad_w_begin': pads[1]})
            mapping.update({'pad_w_end':   pads[3]})
            mapping.update({'stride_h': strides[0]})
            mapping.update({'stride_w': strides[1]})
            mapping.update({'dilation_h': dilations[0]})
            mapping.update({'dilation_w': dilations[1]})
            mapping.update({'storage_order': storage_order})
            mapping.update({'pragma':self.PRAGMA_OMP if self.OpenMP else ''})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})

        elif (ndim == 5):
            TemplateStatements = '''
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_d = {X_d2};
                const int  X_h = {X_d3};
                const int  X_w = {X_d4};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_d = {Y_d2};
                const int  Y_h = {Y_d3};
                const int  Y_w = {Y_d4};
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
                const int  dilation_d = {dilation_d};
                const int  dilation_h = {dilation_h};
                const int  dilation_w = {dilation_w};
                const int  storage_order = {storage_order};

                const int  kernel_shape_d_min = -pad_d_begin;
                const int  kernel_shape_d_max = (kernel_shape_d - pad_d_begin);
                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0][0]) * Y_n * Y_c * Y_d * Y_h * Y_w );

                for (int n=0; n<Y_n; n++) {{
                    {pragma}
                    for (int c=0; c<Y_c; c++) {{
                        for (int d=0; d<Y_d; d++) {{
                            for (int h=0; h<Y_h; h++) {{
                                for (int w=0; w<Y_w; w++) {{
                                    {t} pool;
                                    int  max_flag;
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (int kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                        if ((d*stride_d+kd*dilation_d < 0) || (d*stride_d+kd*dilation_d >= X_d)) {{ continue; }}
                                        for (int kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            if ((h*stride_h+kh*dilation_h < 0) || (h*stride_h+kh*dilation_h >= X_h)) {{ continue; }}
                                            for (int kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                if ((w*stride_w+kw*dilation_w < 0) || (w*stride_w+kw*dilation_w >= X_w)) {{ continue; }}
                                                if (pool < X[n][c][d*stride_d+kd*dilation_d][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w]) {{
                                                    pool = X[n][c][d*stride_d+kd*dilation_d][h*stride_h+kh*dilation_h][w*stride_w+kw*dilation_w];
                                                    max_flag = 1;
                                                }}
                                            }}
                                        }}
                                    }}
                                    if (max_flag) {{
                                        Y[n][c][d][h][w] = pool;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            '''
            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'X_d3': self.input_tensor_shapes[0][3]})
            mapping.update({'X_d4': self.input_tensor_shapes[0][4]})
            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})
            mapping.update({'Y_d3': self.output_tensor_shapes[0][3]})
            mapping.update({'Y_d4': self.output_tensor_shapes[0][4]})
            mapping.update({'kernel_shape_d': kernel_shape[0]})
            mapping.update({'kernel_shape_h': kernel_shape[1]})
            mapping.update({'kernel_shape_w': kernel_shape[2]})
            mapping.update({'pad_d_begin': pads[0]})
            mapping.update({'pad_d_end':   pads[3]})
            mapping.update({'pad_h_begin': pads[1]})
            mapping.update({'pad_h_end':   pads[4]})
            mapping.update({'pad_w_begin': pads[2]})
            mapping.update({'pad_w_end':   pads[5]})
            mapping.update({'stride_d': strides[0]})
            mapping.update({'stride_h': strides[1]})
            mapping.update({'stride_w': strides[2]})
            mapping.update({'dilation_d': dilations[0]})
            mapping.update({'dilation_h': dilations[1]})
            mapping.update({'dilation_w': dilations[2]})
            mapping.update({'storage_order': storage_order})
            mapping.update({'pragma':self.PRAGMA_OMP if self.OpenMP else ''})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})

        # 3        
        TemplateFunction = cleandoc('''
        void {op_func_name}(void *op_param, {t} X{dims_X}, {t} Y{dims}, void *inputs_params, void* outputs_params) {{
            {statements}
        }}
        ''')

        mappingf = {}
        mappingf.update({'op_func_name': self.get_func_name()})
        mappingf.update({'X': self.input_tensor_names[0]})
        mappingf.update({'dims_X': c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
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
    def version_8(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)

    @classmethod
    def version_10(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)
