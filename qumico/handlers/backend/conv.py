from inspect import cleandoc
import math
import numpy as np

from onnx.backend.base import namedtupledict

from qumico.common import c_helper
from qumico.common import data_type
from qumico.device import RaspberryPi3, QumicoDeviceType, QumicoDevice
from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op


@onnx_op('Conv')
class Conv(BackendHandler):
    PARALLEL_SIZE = 1
    WORK_PAD_I_SIZE = 0
    WORK_PAD_W_SIZE = 0
    WORK_PAD_O_SIZE = 0

    SIMD = False
    OpenMP = False

    @classmethod
    def instantiate(cls, node, **kwargs):

        device = kwargs.get("device")
        if device.__class__ == RaspberryPi3 and QumicoDeviceType.ARMNeon in device.options:
            cls.PARALLEL_SIZE = device.PARALELL_SIZE_NEON
            cls.SIMD = True

        if (issubclass(device.__class__, QumicoDevice) and
                QumicoDeviceType.OpenMP in device.options):
            cls.OpenMP = True

        input_data1 = node.input_tensor[0]
        input_data2 = node.input_tensor[1]

        if (len(node.input_tensor) == 3):
            input_data3 = node.input_tensor[2]
        attrs = node.attrs
        if (input_data1.ndim == 3):
            if (attrs.get('dilations') is None):
                attrs['dilations'] = (1, 1)
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
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0, 0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0, 0)
            #                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2] - 1 - (attrs['kernel_shape'][0] - 1) * attrs['dilations'][0] +
                            attrs['pads'][0] + attrs['pads'][1]) / attrs['strides'][0]) + 1)
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
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.floor(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                    attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                   attrs['kernel_shape'][1] - input_data1.shape[3]) / 2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                   attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.floor(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                    attrs['kernel_shape'][1] - input_data1.shape[3]) / 2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0, 0, 0, 0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0, 0, 0, 0)
            #                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2] - 1 - (attrs['kernel_shape'][0] - 1) * attrs['dilations'][0] +
                            attrs['pads'][0] + attrs['pads'][2]) / attrs['strides'][0]) + 1,
                math.floor((input_data1.shape[3] - 1 - (attrs['kernel_shape'][1] - 1) * attrs['dilations'][1] +
                            attrs['pads'][1] + attrs['pads'][3]) / attrs['strides'][1]) + 1)

            if cls.SIMD:
                work_pad_i = (node.input_tensor[0].shape[0] * node.input_tensor[0].shape[1] *
                              (node.input_tensor[0].shape[2] + attrs['pads'][0] + attrs['pads'][2]) *
                              (node.input_tensor[0].shape[3] + attrs['pads'][1] + attrs['pads'][3]))
                work_pad_w = np.prod(node.input_tensor[1].shape)
                work_pad_o = np.prod(outputs_shape)

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
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.floor(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                    attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.floor(((math.ceil(input_data1.shape[4] / attrs['strides'][2]) - 1) * attrs['strides'][2] +
                                    attrs['kernel_shape'][2] - input_data1.shape[4]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                   attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[4] / attrs['strides'][2]) - 1) * attrs['strides'][2] +
                                   attrs['kernel_shape'][2] - input_data1.shape[4]) / 2)
                    )
                elif (auto_pad == 'SAME_LOWER'):
                    attrs['pads'] = (
                        math.ceil(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                   attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                   attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.ceil(((math.ceil(input_data1.shape[4] / attrs['strides'][2]) - 1) * attrs['strides'][2] +
                                   attrs['kernel_shape'][2] - input_data1.shape[4]) / 2),
                        math.floor(((math.ceil(input_data1.shape[2] / attrs['strides'][0]) - 1) * attrs['strides'][0] +
                                    attrs['kernel_shape'][0] - input_data1.shape[2]) / 2),
                        math.floor(((math.ceil(input_data1.shape[3] / attrs['strides'][1]) - 1) * attrs['strides'][1] +
                                    attrs['kernel_shape'][1] - input_data1.shape[3]) / 2),
                        math.floor(((math.ceil(input_data1.shape[4] / attrs['strides'][2]) - 1) * attrs['strides'][2] +
                                    attrs['kernel_shape'][2] - input_data1.shape[4]) / 2)
                    )
                elif (auto_pad == 'VALID'):
                    attrs['pads'] = (0, 0, 0, 0, 0, 0)
                elif (auto_pad == 'NOTSET' or auto_pad is None):
                    attrs['pads'] = (0, 0, 0, 0, 0, 0)
            #                    raise ValueError()

            outputs_shape = (
                input_data1.shape[0],
                input_data2.shape[0],
                math.floor((input_data1.shape[2] - 1 - (attrs['kernel_shape'][0] - 1) * attrs['dilations'][0] +
                            attrs['pads'][0] + attrs['pads'][3]) / attrs['strides'][0]) + 1,
                math.floor((input_data1.shape[3] - 1 - (attrs['kernel_shape'][1] - 1) * attrs['dilations'][1] +
                            attrs['pads'][1] + attrs['pads'][4]) / attrs['strides'][1]) + 1,
                math.floor((input_data1.shape[4] - 1 - (attrs['kernel_shape'][2] - 1) * attrs['dilations'][2] +
                            attrs['pads'][2] + attrs['pads'][5]) / attrs['strides'][2]) + 1)
        else:
            raise (ValueError)
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
            """
            typedef struct {{
                char* name;
                int   dilations[4];
                int   group;
                int   kernel_shape[2];
                int   pads[4];
                int   strides[2];
            }} ConvOpParam;
            """
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_op_variale_def(cls):
        res = "#define SIMD_VECTOR_SIZE ({})".format(str(cls.PARALLEL_SIZE)) + "\n"
        res += "\n"
        res += "#ifndef ALIGN_SIZE" + "\n"
        res += "#define ALIGN_SIZE (sizeof(float) * SIMD_VECTOR_SIZE)" + "\n"
        res += "#endif" + "\n"
        res += "\n"
        res += "__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_i[{}];".format(str(cls.WORK_PAD_I_SIZE)) + "\n"
        res += "__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_w[{}];".format(str(cls.WORK_PAD_W_SIZE)) + "\n"
        res += "__attribute__ ((aligned(ALIGN_SIZE))) float work_pad_o[{}];".format(str(cls.WORK_PAD_O_SIZE)) + "\n"
        res += "\n"
        res += "#define mat_idx4(a, a_max, b, b_max, c, c_max, d, d_max) ((a)*(b_max)*(c_max)*(d_max) +(b)*(c_max)*(d_max) +(c)*(d_max) +(d))" + "\n"
        res += "#define mat_idx5(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max) +(b)*(c_max)*(d_max)*(e_max) +(c)*(d_max)*(e_max) +(d)*(e_max) +(e))" + "\n"

        return res

    def generate_c_code(self, **kwargs):
        res = ''
        res += '\n'.join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res += '\n\n'

        # param type
        res += self.get_c_param_type()
        res += '\n\n'

        # device
        if self.SIMD:
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

        if (ndim == 3):
            TemplateStatements = """
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_w = {X_d2};
                const int  W_m = {W_d0};
                const int  W_c = {W_d1};
                const int  W_kw = {W_d2};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_w = {Y_d2};
                const int  B_n = {B_d0};
                const int  dilation_w = {dilation_w};
                const int  kernel_shape_w = {kernel_shape_w};
                const int  pad_w_begin = {pad_w_begin};
                const int  pad_w_end = {pad_w_end};
                const int  stride_w = {stride_w};

                int  n;
                int  w;
                int  kw;
                int  ic, oc;
                int  current_w;

                const int  kernel_shape_w_min = 0;
                const int  kernel_shape_w_max = {kernel_shape_w};

#if !{B_d0} // B is None.
                memset( (void *)Y, 0, sizeof(Y[0][0][0]) * Y_n * Y_c * Y_w );
#endif // B
                for (n=0; n<Y_n; n++) {{
#if {B_d0} // B has elements.
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (w=0; w<Y_w; w++) {{
                            Y[n][oc][w] = B[oc];
                        }}
                    }}
#endif // B
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (ic=0; ic<X_c; ic++) {{
                            for (w=0; w<Y_w; w++) {{
                                for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                    current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                    if (current_w<0 || current_w>=X_w) {{ continue; }}
                                    Y[n][oc][w] += X[n][ic][current_w]
                                                    * W[oc][ic][kw];
                                }}
                            }}
                        }}
                    }}
                }}
            """

            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'W_d0': self.input_tensor_shapes[1][0]})
            mapping.update({'W_d1': self.input_tensor_shapes[1][1]})
            mapping.update({'W_d2': self.input_tensor_shapes[1][2]})
            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})

            if (len(self.input_tensor) == 3):
                mapping.update({'B_d0': self.input_tensor_shapes[2][0]})
            else:
                mapping.update({'B_d0': 0})
            mapping.update({'dilation_w': dilations[0]})
            mapping.update({'group': group})
            mapping.update(
                {'kernel_shape_w': kernel_shape[0] if (kernel_shape[0] != 0) else self.input_tensor_shapes[1][2]})
            mapping.update({'pad_w_begin': pads[0]})
            mapping.update({'pad_w_end': pads[1]})
            mapping.update({'stride_w': strides[0]})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        elif (ndim == 4):
            TemplateStatements = """
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_h = {X_d2};
                const int  X_w = {X_d3};
                const int  aligned_X_c = {aligned_ic};
                const int  padded_X_h = {X_d2}+{pad_h_begin}+{pad_h_end};
                const int  padded_X_w = {X_d3}+{pad_w_begin}+{pad_w_end};
                const int  W_m = {W_d0};
                const int  W_c = {W_d1};
                const int  W_kh = {W_d2};
                const int  W_kw = {W_d3};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_h = {Y_d2};
                const int  Y_w = {Y_d3};
                const int  aligned_Y_c = {aligned_oc};
                const int  padded_Y_h = {Y_d2}+{pad_h_begin}+{pad_h_end};
                const int  padded_Y_w = {Y_d3}+{pad_w_begin}+{pad_w_end};
                const int  B_n = {B_d0};
                const int  dilation_h = {dilation_h};
                const int  dilation_w = {dilation_w};
                const int  kernel_shape_h = {kernel_shape_h};
                const int  kernel_shape_w = {kernel_shape_w};
                const int  pad_h_begin = {pad_h_begin};
                const int  pad_w_begin = {pad_w_begin};
                const int  pad_h_end = {pad_h_end};
                const int  pad_w_end = {pad_w_end};
                const int  stride_h = {stride_h};
                const int  stride_w = {stride_w};
                const int  group = {group};

                int  n;
                int  h, w;
                int  kh, kw;
                int  ic, oc;
                int  oc1, oc2;
                int  current_h, current_w;

                const int  kernel_shape_h_min = 0;
                const int  kernel_shape_h_max = {kernel_shape_h};
                const int  kernel_shape_w_min = 0;
                const int  kernel_shape_w_max = {kernel_shape_w};
            """

            simd_vector_size = self.PARALLEL_SIZE
            aligned_ic = math.ceil(self.input_tensor_shapes[0][1] / simd_vector_size) * simd_vector_size
            aligned_oc = math.ceil(self.output_tensor_shapes[0][1] / simd_vector_size) * simd_vector_size
            kernel_shape_h = kernel_shape[0] if (kernel_shape[0] != 0) else self.input_tensor_shapes[1][2]
            kernel_shape_w = kernel_shape[1] if (kernel_shape[1] != 0) else self.input_tensor_shapes[1][3]
            use_simd = 0
            if (self.SIMD and (kernel_shape_h == 3) and (kernel_shape_w == 3) and (dilations[0] == 1) and (
                    dilations[1] == 1) and ((self.output_tensor_shapes[0][1] % 4) == 0)):
                use_simd = 1
                TemplateStatements += """
// padding & transpose input data : start
#ifdef TRANSPOSE
                    for (n=0; n<X_n; n++) {{
#pragma omp parallel for
                        for (ic=0; ic<aligned_X_c; ic++) {{
                            for (h=0; h<padded_X_h; h++) {{
                                for (w=0; w<padded_X_w; w++) {{
                                    work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, h, (padded_X_h), w, (padded_X_w))] = 0.0f;
                                }}
                            }}
                        }}
                    }}
                    for (n=0; n<X_n; n++) {{
#pragma omp parallel for
                        for (ic=0; ic<X_c; ic++) {{
                            for (h=0; h<X_h; h++) {{
                                for (w=0; w<X_w; w++) {{
                                    work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+pad_h_begin), (padded_X_h), (w+pad_w_begin), (padded_X_w))] = X[n][ic][h][w];
                                }}
                            }}
                        }}
                    }}
// padding & transpose input data : end
#endif  // TRANSPOSE
                """

                TemplateStatements += """
// transpose weight : start
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (kh=0; kh<kernel_shape_h; kh++) {{
                    for (kw=0; kw<kernel_shape_w; kw++) {{
                        for (ic=0; ic<aligned_X_c; ic++) {{
                            for (oc=0; oc<aligned_Y_c; oc++) {{
                                    work_pad_w[mat_idx4(kh, kernel_shape_h, kw, kernel_shape_w, ic, aligned_X_c, oc, aligned_Y_c)] = W[oc][ic][kh][kw];
                            }}
                        }}
                    }}
                }}

#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<aligned_Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (kh=0; kh<kernel_shape_h; kh++) {{
                            for (kw=0; kw<kernel_shape_w; kw++) {{
                                for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                    work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, kh, kernel_shape_h, kw, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)] = W[oc1*SIMD_VECTOR_SIZE+oc2][ic][kh][kw];
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE
// padding & transpose weight : end
                """

                TemplateStatements += """
#ifndef TRANSPOSE
// no transpose
                for (n=0; n<X_n; n++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (w=0; w<padded_X_w; w++) {{
                            work_pad_i[mat_idx4(n, X_n, ic, X_c, 0, X_h, w, X_w)] = 0.0f;
                            work_pad_i[mat_idx4(n, X_n, ic, X_c, (X_h+pad_h_begin), X_h, w, X_w)] = 0.0f;
                        }}
                        for (h=0; h<X_h; h++) {{
                            work_pad_i[mat_idx4(n, X_n, ic, X_c, (h+pad_h_begin), X_h, 0, X_w)] = 0.0f;
                            work_pad_i[mat_idx4(n, X_n, ic, X_c, (h+pad_h_begin), X_h, (X_w+pad_w_begin), X_w)] = 0.0f;
                            for (w=0; w<X_w; w++) {{
                                work_pad_i[mat_idx4(n, X_n, ic, X_c, (h+pad_h_begin), X_h, (w+pad_w_begin), X_w)] = X[n][ic][h][w];
                            }}
                        }}
                    }}
                }}
#if !{B_d0} // Bias is None.
                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * (padded_Y_h) * (padded_Y_w) );
#endif // Bias
                for (n=0; n<Y_n; n++) {{
#if {B_d0} // Bias has elements.
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                Y[n][oc][h][w] = B[oc];
                            }}
                        }}
                    }}
#endif // Bias
#pragma omp parallel for
                    """
                if group <= 1:
                    TemplateStatements += """ 
                    for (oc=0; oc<Y_c; oc++) {{
                    """
                else:
                    pass
                TemplateStatements += """
                        for (ic=0; ic<X_c; ic++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                        current_h = h*stride_h+kh*dilation_h;
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                            current_w = w*stride_w+kw*dilation_w;
                """
                if group <= 1:
                    TemplateStatements += """
                                            Y[n][oc][h][w] += work_pad_i[mat_idx4(n, X_n, ic, X_c, current_h, X_h, current_w, X_w)] * W[oc][ic][kh][kw];
                """
                else:
                    TemplateStatements += """
                                            Y[n][ic][h][w] += work_pad_i[mat_idx4(n, X_n, ic, X_c, current_h, X_h, current_w, X_w)] * W[ic][ic/group][kh][kw];
                """

                TemplateStatements += """
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                """
                if group <= 1:
                    TemplateStatements += """
                }}
                """
                else:
                    pass
                TemplateStatements += """
#endif   // !TRANSPOSE
                """

                TemplateStatements += """
#ifdef TRANSPOSE
#if !{B_d0} // Bias is None.
            memset( (void *)work_pad_o, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * (padded_Y_h) * (padded_Y_w) );
#endif // Bias
            for (n=0; n<Y_n; n++) {{
#if {B_d0} // Bias has elements.
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (h=0; h<Y_h; h++) {{
                    for (w=0; w<Y_w; w++) {{
                        for (oc=0; oc<Y_c; oc++) {{
                            work_pad_o[mat_idx4(n, Y_n, h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc, Y_c)] = B[oc];
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<aligned_Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (h=0; h<Y_h; h++) {{
                        for (w=0; w<Y_w; w++) {{
                            for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                work_pad_o[mat_idx5(n, Y_n, oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] = B[oc1*SIMD_VECTOR_SIZE+oc2];
                            }}
                        }}
                    }}
                }}
#endif // TRANSPOSE TYPE
#endif // Bias
                """

                TemplateStatements += """
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (h=0; h<Y_h; h++) {{
                    for (w=0; w<Y_w; w++) {{
                        for (ic=0; ic<X_c; ic++) {{
                            for (oc=0; oc<Y_c; oc++) {{
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+0), (X_h+2), (w+0), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(0, kernel_shape_h, 0, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+0), (X_h+2), (w+1), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(0, kernel_shape_h, 1, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+0), (X_h+2), (w+2), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(0, kernel_shape_h, 2, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+1), (X_h+2), (w+0), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(1, kernel_shape_h, 0, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+1), (X_h+2), (w+1), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(1, kernel_shape_h, 1, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+1), (X_h+2), (w+2), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(1, kernel_shape_h, 2, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+2), (X_h+2), (w+0), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(2, kernel_shape_h, 0, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+2), (X_h+2), (w+1), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(2, kernel_shape_h, 1, kernel_shape_w, ic, X_c, oc, Y_c)];
                                work_pad_o[mat_idx4(n, Y_n, h, Y_h, w, Y_w, oc, Y_c)] += work_pad_i[mat_idx4(n, X_n, (h+2), (X_h+2), (w+2), (X_w+2), ic, aligned_X_c)] * work_pad_w[mat_idx4(2, kernel_shape_h, 2, kernel_shape_w, ic, X_c, oc, Y_c)];
                            }}
                        }}
                    }}
                }}
            }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                    for (ic=0; ic<X_c; ic++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+0), (padded_X_h), (w+0), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 0, kernel_shape_h, 0, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+0), (padded_X_h), (w+1), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 0, kernel_shape_h, 1, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+0), (padded_X_h), (w+2), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 0, kernel_shape_h, 2, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+1), (padded_X_h), (w+0), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 1, kernel_shape_h, 0, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+1), (padded_X_h), (w+1), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 1, kernel_shape_h, 1, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+1), (padded_X_h), (w+2), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 1, kernel_shape_h, 2, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+2), (padded_X_h), (w+0), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 2, kernel_shape_h, 0, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+2), (padded_X_h), (w+1), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 2, kernel_shape_h, 1, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                    work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)] += work_pad_i[mat_idx4(n, X_n, ic, aligned_X_c, (h+2), (padded_X_h), (w+2), (padded_X_w))] * work_pad_w[mat_idx5(oc1, (aligned_Y_c/SIMD_VECTOR_SIZE), ic, aligned_X_c, 2, kernel_shape_h, 2, kernel_shape_w, oc2, SIMD_VECTOR_SIZE)];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
#endif  // TRANSPOSE TYPE
                    """

                TemplateStatements += """
// transpose start
#if TRANSPOSE == 1  // M_MINOR
#pragma omp parallel for
                for (n=0; n<Y_n; n++) {{
                    for (oc=0; oc<Y_c; oc++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                Y[n][oc][h][w] = work_pad_o[mat_idx4(n, Y_n, h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc, Y_c)];
                            }}
                        }}
                    }}
                }}
#elif TRANSPOSE == 2  // M_SEPARATE
#pragma omp parallel for
                for (n=0; n<Y_n; n++) {{
                    for (oc1=0; oc1<Y_c/SIMD_VECTOR_SIZE; oc1++) {{
                        for (oc2=0; oc2<SIMD_VECTOR_SIZE; oc2++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    Y[n][oc1*SIMD_VECTOR_SIZE+oc2][h][w] = work_pad_o[mat_idx5(n, Y_n, oc1, (Y_c/SIMD_VECTOR_SIZE), h+pad_h_begin, (padded_Y_h), w+pad_w_begin, (padded_Y_w), oc2, SIMD_VECTOR_SIZE)];
                                }}
                            }}
                        }}
                    }}
                }}
#endif  // TRANSPOSE TYPE
#endif  // TRANSPOSE
// transpose end

                """

            else:  # CPU
                TemplateStatements += """
#if !{B_d0} // Bias is None.
                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );
#endif // B
                for (n=0; n<Y_n; n++) {{
#if {B_d0} // Bias has elements.
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (h=0; h<Y_h; h++) {{
                            for (w=0; w<Y_w; w++) {{
                                Y[n][oc][h][w] = B[oc];
                            }}
                        }}
                    }}
#endif // Bias
#pragma omp parallel for """

                if group <= 1:
                    TemplateStatements += """
                    for (oc=0; oc<Y_c; oc++) {{
                    """
                else:
                    pass

                TemplateStatements += """
                        for (ic=0; ic<X_c; ic++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) {{ continue; }}
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) {{ continue; }}

                  """
                if group <= 1:
                    TemplateStatements += """
                                            Y[n][oc][h][w] += X[n][ic][current_h][current_w]
                                                            * W[oc][ic][kh][kw];
                """
                else:
                    TemplateStatements += """
                                            Y[n][ic][h][w] += X[n][ic][current_h][current_w]
                                                            * W[ic][ic/group][kh][kw];
                """

                TemplateStatements += """
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                """
                if group <= 1:
                    TemplateStatements += """
                }}
                """
                else:
                    pass

            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'X_d3': self.input_tensor_shapes[0][3]})
            mapping.update({'W_d0': self.input_tensor_shapes[1][0]})
            mapping.update({'W_d1': self.input_tensor_shapes[1][1]})
            mapping.update({'W_d2': self.input_tensor_shapes[1][2]})
            mapping.update({'W_d3': self.input_tensor_shapes[1][3]})
            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})
            mapping.update({'Y_d3': self.output_tensor_shapes[0][3]})
            mapping.update({'aligned_ic': aligned_ic})
            mapping.update({'aligned_oc': aligned_oc})

            if (len(self.input_tensor) == 3):
                mapping.update({'B_d0': self.input_tensor_shapes[2][0]})
            else:
                mapping.update({'B_d0': 0})
            mapping.update({'dilation_h': dilations[0]})
            mapping.update({'dilation_w': dilations[1]})
            mapping.update({'group': group})
            mapping.update(
                {'kernel_shape_h': kernel_shape[0] if (kernel_shape[0] != 0) else self.input_tensor_shapes[1][2]})
            mapping.update(
                {'kernel_shape_w': kernel_shape[1] if (kernel_shape[1] != 0) else self.input_tensor_shapes[1][3]})
            mapping.update({'pad_h_begin': pads[0]})
            mapping.update({'pad_h_end': pads[2]})
            mapping.update({'pad_w_begin': pads[1]})
            mapping.update({'pad_w_end': pads[3]})
            mapping.update({'padded_size_h': self.input_tensor_shapes[0][2] + pads[0] + pads[2]})
            mapping.update({'padded_size_w': self.input_tensor_shapes[0][3] + pads[1] + pads[3]})
            mapping.update({'stride_h': strides[0]})
            mapping.update({'stride_w': strides[1]})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})
        else:  # ndim=5
            TemplateStatements = """
                const int  X_n = {X_d0};
                const int  X_c = {X_d1};
                const int  X_d = {X_d2};
                const int  X_h = {X_d3};
                const int  X_w = {X_d4};
                const int  W_m = {W_d0};
                const int  W_c = {W_d1};
                const int  W_kd = {W_d2};
                const int  W_kh = {W_d3};
                const int  W_kw = {W_d4};
                const int  Y_n = {Y_d0};
                const int  Y_c = {Y_d1};
                const int  Y_d = {Y_d2};
                const int  Y_h = {Y_d3};
                const int  Y_w = {Y_d4};
                const int  B_n = {B_d0};
                const int  dilation_d = {dilation_d};
                const int  dilation_h = {dilation_h};
                const int  dilation_w = {dilation_w};
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
                int  current_d, current_h, current_w;

                const int  kernel_shape_d_min = 0;
                const int  kernel_shape_d_max = {kernel_shape_d};
                const int  kernel_shape_h_min = 0;
                const int  kernel_shape_h_max = {kernel_shape_h};
                const int  kernel_shape_w_min = 0;
                const int  kernel_shape_w_max = {kernel_shape_w};

#if !{B_d0} // B is None.
                memset( (void *)Y, 0, sizeof(Y[0][0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );
#endif // B
                for (n=0; n<Y_n; n++) {{
#if {B_d0} // B has elements.
#pragma omp parallel for
                    for (oc=0; oc<Y_c; oc++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    Y[n][oc][d][h][w] = B[oc];
                                }}
                            }}
                        }}
                    }}
#endif // B
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
                                                    Y[n][oc][d][h][w] += X[n][ic][current_d][current_h][current_w]
                                                                       * W[oc][ic][kd][kh][kw];
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            """

            mapping = {}
            mapping.update({'X_d0': self.input_tensor_shapes[0][0]})
            mapping.update({'X_d1': self.input_tensor_shapes[0][1]})
            mapping.update({'X_d2': self.input_tensor_shapes[0][2]})
            mapping.update({'X_d3': self.input_tensor_shapes[0][3]})
            mapping.update({'X_d4': self.input_tensor_shapes[0][4]})
            mapping.update({'W_d0': self.input_tensor_shapes[1][0]})
            mapping.update({'W_d1': self.input_tensor_shapes[1][1]})
            mapping.update({'W_d2': self.input_tensor_shapes[1][2]})
            mapping.update({'W_d3': self.input_tensor_shapes[1][3]})
            mapping.update({'W_d4': self.input_tensor_shapes[1][4]})

            mapping.update({'Y_d0': self.output_tensor_shapes[0][0]})
            mapping.update({'Y_d1': self.output_tensor_shapes[0][1]})
            mapping.update({'Y_d2': self.output_tensor_shapes[0][2]})
            mapping.update({'Y_d3': self.output_tensor_shapes[0][3]})
            mapping.update({'Y_d4': self.output_tensor_shapes[0][4]})
            if (len(self.input_tensor) == 3):
                mapping.update({'B_d0': self.input_tensor_shapes[2][0]})
            else:
                mapping.update({'B_d0': 0})
            mapping.update({'dilation_d': dilations[0]})
            mapping.update({'dilation_h': dilations[1]})
            mapping.update({'dilation_w': dilations[2]})
            mapping.update({'group': group})
            mapping.update(
                {'kernel_shape_d': kernel_shape[0] if (kernel_shape[0] != 0) else self.input_tensor_shapes[1][2]})
            mapping.update(
                {'kernel_shape_h': kernel_shape[1] if (kernel_shape[1] != 0) else self.input_tensor_shapes[1][3]})
            mapping.update(
                {'kernel_shape_w': kernel_shape[2] if (kernel_shape[2] != 0) else self.input_tensor_shapes[1][4]})
            mapping.update({'pad_d_begin': pads[0]})
            mapping.update({'pad_d_end': pads[3]})
            mapping.update({'pad_h_begin': pads[1]})
            mapping.update({'pad_h_end': pads[4]})
            mapping.update({'pad_w_begin': pads[2]})
            mapping.update({'pad_w_end': pads[5]})
            mapping.update({'stride_d': strides[0]})
            mapping.update({'stride_h': strides[1]})
            mapping.update({'stride_w': strides[2]})
            mapping.update({'t': data_type.np2c(self.output_tensor_dtypes[0])})

        # 3
        if (use_simd == 1):
            TemplateFunction = """
//#define TRANSPOSE 1  // M_MINOR
#define TRANSPOSE 2  // M_SEPARATE
//#undef TRANSPOSE
            """
        else:
            TemplateFunction = """
#undef TRANSPOSE
            """
        if (len(self.input_tensor) == 3):
            TemplateFunction += cleandoc("""
            void {op_func_name}(void *op_param, {t} X{dims_X}, {t} W{dims_W}, {t} B{dims_B}, {t} Y{dims}, void *inputs_params, void* outputs_params)
            {{
                {statements}
            }}
            """)
        else:
            TemplateFunction += cleandoc("""
            void {op_func_name}(void *op_param, {t} X{dims_X}, {t} W{dims_W}, {t} Y{dims}, void *inputs_params, void* outputs_params)
            {{
                {statements}
            }}
            """)

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
        TemplateVariavbles = cleandoc("""
            int OpShapeNode{node_num}[] = {{{shape}}};
            int OutputShapeNode{node_num}[] = {{{shape}}};
            """)
        ndim = self.output_tensor_ndims[0]
        shape = self.output_tensor_shapes[0]
        mapping = {}
        mapping.update({'shape': ','.join(map(str, shape[:ndim]))})
        mapping.update({'node_num': str(node_num)})

        return TemplateVariavbles.format(**mapping)

    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc = cleandoc("""
        {indent}// define input & output
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
        {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        """)

        mapping = {}
        mapping.update({'node_param_name': node.node_param_name})
        mapping.update({'node_num': str(node_num)})
        mapping.update({'add_name': self.get_name()})
        mapping.update({'ndim': str(self.output_tensor_ndims[0])})
        mapping.update({'output_val_name': self.output_tensor_names[0]})
        mapping.update({'indent': ' ' * indent})

        return TemplateInitFunc.format(**mapping)

    @classmethod
    def need_c_headers(cls):
        return ['string.h']

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)






