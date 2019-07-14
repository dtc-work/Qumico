import string
from inspect import cleandoc
from collections import OrderedDict
import math

import numpy as np

from onnx.backend.base import namedtupledict

from qumico.handlers.backend_handler import BackendHandler
from qumico.handlers.handler import onnx_op
from qumico.common import c_helper
from qumico.common import data_type

@onnx_op("AveragePool")

class AveragePool(BackendHandler):

    @classmethod
    def instantiate(cls, node, **kwargs):
        input_data1 = node.input_tensor[0]
        attrs = node.attrs

        if (input_data1.ndim == 3):
            if (attrs.get("strides") == None):
                attrs["strides"] = (1,)

    # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get("auto_pad")
            if (attrs.get("pads") == None):
                if (auto_pad == "SAME_UPPER"):
                    attrs["pads"] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                    )
                elif (auto_pad == "SAME_LOWER"):
                    attrs["pads"] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                    )
                elif (auto_pad == "VALID"):
                    attrs["pads"] = (0,0)
                elif (auto_pad == "NOTSET" or auto_pad == None):
                    attrs["pads"] = (0,0)
            if (attrs.get("storage_order") == None):
                attrs["storage_order"] = 0
            if (attrs.get("count_include_pad") == None):
                attrs["count_include_pad"] = 0

    # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-1):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == "SAME_UPPER") or (auto_pad == "SAME_LOWER"):
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs["strides"][-1]))
            elif (auto_pad == "VALID"):
                tmp_shape.append(math.ceil((input_data1.shape[-1] - attrs["kernel_shape"][-1] +1)/attrs["strides"][-1]))
            else:
                tmp_shape.append(math.floor((input_data1.shape[-1] + attrs["pads"][0] + attrs["pads"][-1] - attrs["kernel_shape"][-1])/attrs["strides"][-1] +1))
        elif (input_data1.ndim == 4):
            if (attrs.get("strides") == None):
                attrs["strides"] = (1,1)

    # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get("auto_pad")
            if (attrs.get("pads") == None):
                if (auto_pad == "SAME_UPPER"):
                    attrs["pads"] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2)
                    )
                elif (auto_pad == "SAME_LOWER"):
                    attrs["pads"] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2)
                    )
                elif (auto_pad == "VALID"):
                    attrs["pads"] = (0,0,0,0)
                elif (auto_pad == "NOTSET" or auto_pad == None):
                    attrs["pads"] = (0,0,0,0)
            if (attrs.get("storage_order") == None):
                attrs["storage_order"] = 0
            if (attrs.get("count_include_pad") == None):
                attrs["count_include_pad"] = 0

    # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-2):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == "SAME_UPPER") or (auto_pad == "SAME_LOWER"):
                tmp_shape.append(math.ceil(input_data1.shape[-2]/attrs["strides"][-2]))
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs["strides"][-1]))
            elif (auto_pad == "VALID"):
                tmp_shape.append(math.ceil((input_data1.shape[-2] - attrs["kernel_shape"][-2] +1)/attrs["strides"][-2]))
                tmp_shape.append(math.ceil((input_data1.shape[-1] - attrs["kernel_shape"][-1] +1)/attrs["strides"][-1]))
            else:
                tmp_shape.append(math.floor((input_data1.shape[-2] + attrs["pads"][0] + attrs["pads"][-2] - attrs["kernel_shape"][-2])/attrs["strides"][-2] +1))
                tmp_shape.append(math.floor((input_data1.shape[-1] + attrs["pads"][1] + attrs["pads"][-1] - attrs["kernel_shape"][-1])/attrs["strides"][-1] +1))
        elif (input_data1.ndim == 5):
            if (attrs.get("strides") == None):
                attrs["strides"] = (1,1,1)

    # pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
            auto_pad = attrs.get("auto_pad")
            if (attrs.get("pads") == None):
                if (auto_pad == "SAME_UPPER"):
                    attrs["pads"] = (
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs["strides"][2]) -1) * attrs["strides"][2] + attrs["kernel_shape"][2] - input_data1.shape[4])/2),
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs["strides"][2]) -1) * attrs["strides"][2] + attrs["kernel_shape"][2] - input_data1.shape[4])/2)
                    )
                elif (auto_pad == "SAME_LOWER"):
                    attrs["pads"] = (
                        math.ceil(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.ceil(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.ceil(((math.ceil(input_data1.shape[4]/attrs["strides"][2]) -1) * attrs["strides"][2] + attrs["kernel_shape"][2] - input_data1.shape[4])/2),
                        math.floor(((math.ceil(input_data1.shape[2]/attrs["strides"][0]) -1) * attrs["strides"][0] + attrs["kernel_shape"][0] - input_data1.shape[2])/2),
                        math.floor(((math.ceil(input_data1.shape[3]/attrs["strides"][1]) -1) * attrs["strides"][1] + attrs["kernel_shape"][1] - input_data1.shape[3])/2),
                        math.floor(((math.ceil(input_data1.shape[4]/attrs["strides"][2]) -1) * attrs["strides"][2] + attrs["kernel_shape"][2] - input_data1.shape[4])/2)
                    )
                elif (auto_pad == "VALID"):
                    attrs["pads"] = (0,0,0,0,0,0)
                elif (auto_pad == "NOTSET" or auto_pad == None):
                    attrs["pads"] = (0,0,0,0,0,0)
            if (attrs.get("storage_order") == None):
                attrs["storage_order"] = 0
            if (attrs.get("count_include_pad") == None):
                attrs["count_include_pad"] = 0

    # SAME: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    # VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    # NOTSET: output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
            tmp_shape = []
            for d in range(0, input_data1.ndim-3):
                tmp_shape.append(input_data1.shape[d])
            if (auto_pad == "SAME_UPPER") or (auto_pad == "SAME_LOWER"):
                tmp_shape.append(math.ceil(input_data1.shape[-3]/attrs["strides"][-3]))
                tmp_shape.append(math.ceil(input_data1.shape[-2]/attrs["strides"][-2]))
                tmp_shape.append(math.ceil(input_data1.shape[-1]/attrs["strides"][-1]))
            elif (auto_pad == "VALID"):
                tmp_shape.append(math.ceil((input_data1.shape[-3] - attrs["kernel_shape"][-3] +1)/attrs["strides"][-3]))
                tmp_shape.append(math.ceil((input_data1.shape[-2] - attrs["kernel_shape"][-2] +1)/attrs["strides"][-2]))
                tmp_shape.append(math.ceil((input_data1.shape[-1] - attrs["kernel_shape"][-1] +1)/attrs["strides"][-1]))
            else:
                tmp_shape.append(math.floor((input_data1.shape[-3] + attrs["pads"][0] + attrs["pads"][-3] - attrs["kernel_shape"][-3])/attrs["strides"][-3] +1))
                tmp_shape.append(math.floor((input_data1.shape[-2] + attrs["pads"][1] + attrs["pads"][-2] - attrs["kernel_shape"][-2])/attrs["strides"][-2] +1))
                tmp_shape.append(math.floor((input_data1.shape[-1] + attrs["pads"][2] + attrs["pads"][-1] - attrs["kernel_shape"][-1])/attrs["strides"][-1] +1))
        else:
            raise(ValueError)

        outputs_shape = tuple(tmp_shape)
        outputs_dtype = input_data1.dtype
        outputs_dict = {node.valid_var_name(node.outputs[0]): np.ones(shape=outputs_shape, dtype=outputs_dtype)}
        output_tensor = namedtupledict("output_tensor", outputs_dict.keys())(**outputs_dict)

        return cls(node, input_tensor=node.input_tensor, 
                   output_tensor=output_tensor, attrs=attrs)
    

    @classmethod
    def get_param_type_name(cls):
        return "AveragePoolOpParam"


    @classmethod
    def get_c_op_file_name(cls):
        return ["averagepool.c"]


    @classmethod
    def get_c_op_include_header(cls):
        return ["math.h", "float.h"]
    

    @classmethod
    @BackendHandler.dec_generate_once()
    def get_c_param_type(cls):
        TEMPLATE_STRUCT = cleandoc(
            """
            typedef struct {{
                char* name;
                int   kernel_shape[2];
                int   pads[4];
                int   storage_order;
                int   strides[4];
            }} AveragePoolOpParam;
            """
        )
        mapping = {}

        return TEMPLATE_STRUCT.format(**mapping)


    def generate_c_code(self, **kwargs):
        res =""
        res += "\n".join([c_helper.generate_local_include(h) for h in self.get_c_op_include_header()])
        res +="\n\n"

        # param type
        res += self.get_c_param_type()
        res +="\n\n"

        ndim = self.output_tensor_ndims[0]
        if (ndim != 3 and ndim != 4 and ndim != 5):
            raise ValueError()

        kernel_shape = self.attrs["kernel_shape"]
        pads = self.attrs["pads"]
        storage_order = self.attrs["storage_order"]
        strides = self.attrs["strides"]
        count_include_pad = self.attrs["count_include_pad"]

        if (ndim == 3):
            TemplateStatements = """
                int  X_n = {X_d0};
                int  X_c = {X_d1};
                int  X_w = {X_d2};
                int  Y_n = {Y_d0};
                int  Y_c = {Y_d1};
                int  Y_w = {Y_d2};
                int  kernel_shape_w = {kernel_shape_w};
                int  pad_w_begin = {pad_w_begin};
                int  pad_w_end = {pad_w_end};
                int  stride_w = {stride_w};
                int  storage_order = {storage_order};
                int  count_include_pad = {count_include_pad};

                int  n;
                int  c;
                int  w;
                int  kw;
                int  kernel_shape_w_min;
                int  kernel_shape_w_max;
                {t} pool;
                int  data_cnt;

                kernel_shape_w_min = -pad_w_begin;
                kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0.0, sizeof(Y[0][0][0]) * Y_n * Y_c * Y_w );

                for (n=0; n<Y_n; n++) {{
                    for (c=0; c<Y_c; c++) {{
                        for (w=0; w<Y_w; w++) {{
                            pool = 0.0;
                            data_cnt = 0;
                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) {{ continue; }}
                                pool += X[n][c][w*stride_w+kw];
                                data_cnt++;
                            }}
                            if (data_cnt > 0) {{
                                Y[n][c][w] = pool / data_cnt;
                            }}
                        }}
                    }}
                }}
            """
            mapping = {}
            mapping.update({"X_d0": self.input_tensor_shapes[0][0]})
            mapping.update({"X_d1": self.input_tensor_shapes[0][1]})
            mapping.update({"X_d2": self.input_tensor_shapes[0][2]})
            mapping.update({"Y_d0": self.output_tensor_shapes[0][0]})
            mapping.update({"Y_d1": self.output_tensor_shapes[0][1]})
            mapping.update({"Y_d2": self.output_tensor_shapes[0][2]})
            mapping.update({"kernel_shape_w": kernel_shape[0]})
            mapping.update({"pad_w_begin": pads[0]})
            mapping.update({"pad_w_end":   pads[1]})
            mapping.update({"stride_w": strides[0]})
            mapping.update({"storage_order": storage_order})
            mapping.update({"count_include_pad": count_include_pad})
            mapping.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})

        elif (ndim == 4):
            TemplateStatements = """
                int  X_n = {X_d0};
                int  X_c = {X_d1};
                int  X_h = {X_d2};
                int  X_w = {X_d3};
                int  Y_n = {Y_d0};
                int  Y_c = {Y_d1};
                int  Y_h = {Y_d2};
                int  Y_w = {Y_d3};
                int  kernel_shape_h = {kernel_shape_h};
                int  kernel_shape_w = {kernel_shape_w};
                int  pad_h_begin = {pad_h_begin};
                int  pad_w_begin = {pad_w_begin};
                int  pad_h_end = {pad_h_end};
                int  pad_w_end = {pad_w_end};
                int  stride_h = {stride_h};
                int  stride_w = {stride_w};
                int  storage_order = {storage_order};
                int  count_include_pad = {count_include_pad};

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                int  kernel_shape_h_min;
                int  kernel_shape_h_max;
                int  kernel_shape_w_min;
                int  kernel_shape_w_max;
                {t} pool;
                int  data_cnt;

                kernel_shape_h_min = -pad_h_begin;
                kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                kernel_shape_w_min = -pad_w_begin;
                kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0.0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {{
                    for (c=0; c<Y_c; c++) {{
                        if (storage_order == 0) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    pool = 0.0;
                                    data_cnt = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) {{
                                            if (count_include_pad != 0) {{
                                                data_cnt += kernel_shape_w;
                                            }}
                                            continue;
                                        }}
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) {{ 
                                                if (count_include_pad != 0) {{
                                                    data_cnt++;
                                                }}
                                            }} else {{
                                                pool += X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                data_cnt++;
                                            }}
                                        }}
                                    }}
                                    if (data_cnt > 0) {{
                                        Y[n][c][h][w] = pool / data_cnt;
                                    }}
                                }}
                            }}
                        }} else {{
                            for (w=0; w<Y_w; w++) {{
                                for (h=0; h<Y_h; h++) {{
                                    pool = 0.0;
                                    data_cnt = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) {{
                                            if (count_include_pad != 0) {{
                                                data_cnt++;
                                            }}
                                            continue;
                                        }}
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) {{
                                                if (count_include_pad != 0) {{
                                                    data_cnt++;
                                                }}
                                            }} else {{
                                                pool += X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                data_cnt++;
                                            }}
                                        }}
                                    }}
                                    if (data_cnt > 0) {{
                                        Y[n][c][h][w] = pool / data_cnt;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            """
            mapping = {}
            mapping.update({"X_d0": self.input_tensor_shapes[0][0]})
            mapping.update({"X_d1": self.input_tensor_shapes[0][1]})
            mapping.update({"X_d2": self.input_tensor_shapes[0][2]})
            mapping.update({"X_d3": self.input_tensor_shapes[0][3]})
            mapping.update({"Y_d0": self.output_tensor_shapes[0][0]})
            mapping.update({"Y_d1": self.output_tensor_shapes[0][1]})
            mapping.update({"Y_d2": self.output_tensor_shapes[0][2]})
            mapping.update({"Y_d3": self.output_tensor_shapes[0][3]})
            mapping.update({"kernel_shape_h": kernel_shape[0]})
            mapping.update({"kernel_shape_w": kernel_shape[1]})
            mapping.update({"pad_h_begin": pads[0]})
            mapping.update({"pad_h_end":   pads[2]})
            mapping.update({"pad_w_begin": pads[1]})
            mapping.update({"pad_w_end":   pads[3]})
            mapping.update({"stride_h": strides[0]})
            mapping.update({"stride_w": strides[1]})
            mapping.update({"storage_order": storage_order})
            mapping.update({"count_include_pad": count_include_pad})
            mapping.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})

        elif (ndim == 5):
            TemplateStatements = """
                int  X_n = {X_d0};
                int  X_c = {X_d1};
                int  X_d = {X_d2};
                int  X_h = {X_d3};
                int  X_w = {X_d4};
                int  Y_n = {Y_d0};
                int  Y_c = {Y_d1};
                int  Y_d = {Y_d2};
                int  Y_h = {Y_d3};
                int  Y_w = {Y_d4};
                int  kernel_shape_d = {kernel_shape_d};
                int  kernel_shape_h = {kernel_shape_h};
                int  kernel_shape_w = {kernel_shape_w};
                int  pad_d_begin = {pad_d_begin};
                int  pad_h_begin = {pad_h_begin};
                int  pad_w_begin = {pad_w_begin};
                int  pad_d_end = {pad_d_end};
                int  pad_h_end = {pad_h_end};
                int  pad_w_end = {pad_w_end};
                int  stride_d = {stride_d};
                int  stride_h = {stride_h};
                int  stride_w = {stride_w};
                int  storage_order = {storage_order};
                int  count_include_pad = {count_include_pad};

                int  n;
                int  c;
                int  d, h, w;
                int  kd, kh, kw;
                int  kernel_shape_d_min;
                int  kernel_shape_d_max;
                int  kernel_shape_h_min;
                int  kernel_shape_h_max;
                int  kernel_shape_w_min;
                int  kernel_shape_w_max;
                {t} pool;
                int  data_cnt;

                kernel_shape_d_min = -pad_d_begin;
                kernel_shape_d_max = (kernel_shape_d - pad_d_begin);
                kernel_shape_h_min = -pad_h_begin;
                kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                kernel_shape_w_min = -pad_w_begin;
                kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0.0, sizeof(Y[0][0][0][0][0]) * Y_n * Y_c * Y_d * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {{
                    for (c=0; c<Y_c; c++) {{
                        for (d=0; d<Y_d; d++) {{
                            for (h=0; h<Y_h; h++) {{
                                for (w=0; w<Y_w; w++) {{
                                    pool = 0.0;
                                    data_cnt = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {{
                                        if ((d*stride_d+kd < 0) || (d*stride_d+kd >= X_d)) {{ continue; }}
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {{
                                            if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) {{ continue; }}
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {{
                                                if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) {{ continue; }}
                                                pool += X[n][c][d*stride_d+kd][h*stride_h+kh][w*stride_w+kw];
                                                data_cnt++;
                                            }}
                                        }}
                                    }}
                                    if (data_cnt > 0) {{
                                        Y[n][c][d][h][w] = pool / data_cnt;
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            """
            mapping = {}
            mapping.update({"X_d0": self.input_tensor_shapes[0][0]})
            mapping.update({"X_d1": self.input_tensor_shapes[0][1]})
            mapping.update({"X_d2": self.input_tensor_shapes[0][2]})
            mapping.update({"X_d3": self.input_tensor_shapes[0][3]})
            mapping.update({"X_d4": self.input_tensor_shapes[0][4]})
            mapping.update({"Y_d0": self.output_tensor_shapes[0][0]})
            mapping.update({"Y_d1": self.output_tensor_shapes[0][1]})
            mapping.update({"Y_d2": self.output_tensor_shapes[0][2]})
            mapping.update({"Y_d3": self.output_tensor_shapes[0][3]})
            mapping.update({"Y_d4": self.output_tensor_shapes[0][4]})
            mapping.update({"kernel_shape_d": kernel_shape[0]})
            mapping.update({"kernel_shape_h": kernel_shape[1]})
            mapping.update({"kernel_shape_w": kernel_shape[2]})
            mapping.update({"pad_d_begin": pads[0]})
            mapping.update({"pad_d_end":   pads[3]})
            mapping.update({"pad_h_begin": pads[1]})
            mapping.update({"pad_h_end":   pads[4]})
            mapping.update({"pad_w_begin": pads[2]})
            mapping.update({"pad_w_end":   pads[5]})
            mapping.update({"stride_d": strides[0]})
            mapping.update({"stride_h": strides[1]})
            mapping.update({"stride_w": strides[2]})
            mapping.update({"storage_order": storage_order})
            mapping.update({"count_include_pad": count_include_pad})
            mapping.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})

        # 3        
        TemplateFunction = cleandoc("""
        void {op_func_name}(void *op_param, {t} X{dims_X}, {t} Y{dims}, void *inputs_params, void* outputs_params) {{
            {statements}
        }}
        """)

        mappingf = {}
        mappingf.update({"op_func_name": self.get_func_name()})
        mappingf.update({"X": self.input_tensor_names[0]})
        mappingf.update({"dims_X": c_helper.generate_dim_bracket(self.input_tensor_shapes[0])}) 
        mappingf.update({"Y": self.output_tensor_names[0]})
        mappingf.update({"dims": c_helper.generate_dim_bracket(self.output_tensor_shapes[0])}) 
        mappingf.update({"t": data_type.np2c(self.output_tensor_dtypes[0])})
        mappingf.update({"statements": TemplateStatements.format(**mapping)})
        res += "\n\n"
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
        mapping.update({"shape": ",".join(map(str,shape[:ndim]))})
        mapping.update({"node_num": str(node_num)})

        return TemplateVariavbles.format(**mapping)        


    def gen_init_func(self, node, node_num, indent=4, **kwargs):

        TemplateInitFunc=cleandoc("""
        {indent}// define input & output
        {indent}Nodes[{node_num}].op_param = &{node_param_name};
        {indent}Nodes[{node_num}].outputs = &{output_val_name};
        {indent}Nodes[{node_num}].output_ndim = {ndim};
        {indent}Nodes[{node_num}].output_shape = OutputShapeNode{node_num};
        """)
        
        mapping = {}
        mapping.update({"node_param_name": node.node_param_name})
        mapping.update({"node_num": str(node_num)})
        mapping.update({"add_name": self.get_name()})
        mapping.update({"ndim":str(self.output_tensor_ndims[0])})
        mapping.update({"output_val_name": self.output_tensor_names[0]})
        mapping.update({"indent":" " * indent})

        return TemplateInitFunc.format(**mapping)


    @classmethod
    def version_7(cls, node, **kwargs):
        return cls.instantiate(node, **kwargs)


    
"""
void op_AveragePool_2d ( const op_AveragePool_2d_param *param, const void **inputs, void **outputs ) {
    const TYPE_X *X = *inputs;
    TYPE_Y *Y       = *outputs;

    INT64  X_n = param->i0_0max;
    INT64  X_c = param->i0_1max;
    INT64  X_h = param->i0_2max;
    INT64  X_w = param->i0_3max;
    INT64  Y_n = param->o0_0max;
    INT64  Y_c = param->o0_1max;
    INT64  Y_h = param->o0_2max;
    INT64  Y_w = param->o0_3max;
    INT64  kernel_shape_h = param->kernel_shape_h;
    INT64  kernel_shape_w = param->kernel_shape_w;
    INT64  pad_h_begin    = param->pad_h_begin;
    INT64  pad_w_begin    = param->pad_w_begin;
    INT64  pad_h_end      = param->pad_h_end;
    INT64  pad_w_end      = param->pad_w_end;
    INT64  storage_order  = param->storage_order;
    INT64  stride_h       = param->stride_h;
    INT64  stride_w       = param->stride_w;

    INT64  c;
    INT64  h, w;
    INT64  kh, kw;
    FLOAT  pool;

    for (c=0; c<Y_c; c++) {
        if (storage_order == 0) {
            for (h=0; h<Y_h; h++) {
                for (w=0; w<Y_w; w++) {
                    if (h*stride_h<pad_h_begin || h*stride_h>=(Y_h-pad_h_end) || w*stride_w<pad_w_begin || w*stride_w>=(Y_w-pad_w_end)) {
                        pool = 0.0;
                    } else {
                        pool = -9e+32;
                        for (kh=h*stride_h; kh<h*stride_h+kernel_shape_h; kh++) {
                            for (kw=w*stride_w; kw<w*stride_w+kernel_shape_w; kw++) {
                                if (kh<pad_h_begin || kh>=(Y_h*stride_h-pad_h_end) || kw<pad_w_begin || kw>=(Y_w*stride_w-pad_w_end)) {continue;}
                                if (pool < *(X + c*X_h*X_w + kh*X_w + kw)) {
                                    pool = *(X + c*X_h*X_w + kh*X_w + kw);
                                }
                            }
                        }
                    }
                    *(Y + c*Y_h*Y_w + h*Y_w + w) = pool;
                }
            }
        } else {
            for (w=0; w<Y_w; w++) {
                for (h=0; h<Y_h; h++) {
                    if (h*stride_h<pad_h_begin || h*stride_h>=(Y_h-pad_h_end) || w*stride_w<pad_w_begin || w*stride_w>=(Y_w-pad_w_end)) {
                        pool = 0.0;
                    } else {
                        pool = -9e+32;
                        for (kh=h*stride_h; kh<h*stride_h+kernel_shape_h; kh++) {
                            for (kw=w*stride_w; kw<w*stride_w+kernel_shape_w; kw++) {
                                if (kh<pad_h_begin || kh>=(Y_h*stride_h-pad_h_end) || kw<pad_w_begin || kw>=(Y_w*stride_w-pad_w_end)) {continue;}
                                if (pool < *(X + c*X_h*X_w + kh*X_w + kw)) {
                                    pool = *(X + c*X_h*X_w + kh*X_w + kw);
                                }
                            }
                        }
                    }
                    *(Y + c*Y_w*Y_h + w*Y_h + h) = pool;
                }
            }
        }
    }
}

void op_AveragePool_2d ( const op_AveragePool_2d_param *param, const void **inputs, void **outputs ) {
    const TYPE_X *X = *inputs;
    TYPE_Y *Y       = *outputs;

    INT64  X_n = param->i0_0max;
    INT64  X_c = param->i0_1max;
    INT64  X_h = param->i0_2max;
    INT64  X_w = param->i0_3max;
    INT64  Y_n = param->o0_0max;
    INT64  Y_c = param->o0_1max;
    INT64  Y_h = param->o0_2max;
    INT64  Y_w = param->o0_3max;
    INT64  kernel_shape_h = param->kernel_shape_h;
    INT64  kernel_shape_w = param->kernel_shape_w;
    INT64  pad_h_begin    = param->pad_h_begin;
    INT64  pad_w_begin    = param->pad_w_begin;
    INT64  pad_h_end      = param->pad_h_end;
    INT64  pad_w_end      = param->pad_w_end;
    INT64  storage_order  = param->storage_order;
    INT64  stride_h       = param->stride_h;
    INT64  stride_w       = param->stride_w;

    INT64  c;
    INT64  h, w;
    INT64  kh, kw;
    FLOAT  pool;

    for (c=0; c<Y_c; c++) {
        if (storage_order == 0) {
            for (h=0; h<Y_h; h++) {
                for (w=0; w<Y_w; w++) {
                    if (h*stride_h<pad_h_begin || h*stride_h>=(Y_h-pad_h_end) || w*stride_w<pad_w_begin || w*stride_w>=(Y_w-pad_w_end)) {
                        pool = 0.0;
                    } else {
                        pool = 0.0;
                        for (kh=h*stride_h; kh<h*stride_h+kernel_shape_h; kh++) {
                            for (kw=w*stride_w; kw<w*stride_w+kernel_shape_w; kw++) {
                                if (kh<pad_h_begin || kh>=(Y_h*stride_h-pad_h_end) || kw<pad_w_begin || kw>=(Y_w*stride_w-pad_w_end)) {continue;}
                                    pool += *(X + c*X_h*X_w + kh*X_w + kw);
                                }
                            }
                        }
                    }
                    *(Y + c*Y_h*Y_w + h*Y_w + w) = pool/(kernel_shape_h * kernel_shape_w);
                }
            }
        } else {
            for (w=0; w<Y_w; w++) {
                for (h=0; h<Y_h; h++) {
                    if (h*stride_h<pad_h_begin || h*stride_h>=(Y_h-pad_h_end) || w*stride_w<pad_w_begin || w*stride_w>=(Y_w-pad_w_end)) {
                        pool = 0.0;
                    } else {
                        pool = 0.0;
                        for (kh=h*stride_h; kh<h*stride_h+kernel_shape_h; kh++) {
                            for (kw=w*stride_w; kw<w*stride_w+kernel_shape_w; kw++) {
                                if (kh<pad_h_begin || kh>=(Y_h*stride_h-pad_h_end) || kw<pad_w_begin || kw>=(Y_w*stride_w-pad_w_end)) {continue;}
                                    pool += *(X + c*X_h*X_w + kh*X_w + kw);
                                }
                            }
                        }
                    }
                    *(Y + c*Y_w*Y_h + w*Y_h + h) = pool/(kernel_shape_h * kernel_shape_w);
                }
            }
        }
    }
}
"""



