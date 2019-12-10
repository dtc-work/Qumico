/**
 * Copyright (c) 2019 Pasona Tech Inc. http://pasonatech.co.jp
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or  substantial portions of
 * the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Date Created : 2019-07-30 15:59:36
 * Version      : dev201907
 */

#include "math.h"
#include "float.h"

typedef struct {
    char* name;
    int   kernel_shape[2];
    int   pads[4];
    int   storage_order;
    int   strides[4];
} AveragePoolOpParam;



void OpAveragePool1(void *op_param, float X[1][256][4][4], float Y[1][256][1][1], void *inputs_params, void* outputs_params) {
    
                int  X_n = 1;
                int  X_c = 256;
                int  X_h = 4;
                int  X_w = 4;
                int  Y_n = 1;
                int  Y_c = 256;
                int  Y_h = 1;
                int  Y_w = 1;
                int  kernel_shape_h = 4;
                int  kernel_shape_w = 4;
                int  pad_h_begin = 0;
                int  pad_w_begin = 0;
                int  pad_h_end = 0;
                int  pad_w_end = 0;
                int  stride_h = 2;
                int  stride_w = 2;
                int  storage_order = 0;
                int  count_include_pad = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                int  kernel_shape_h_min;
                int  kernel_shape_h_max;
                int  kernel_shape_w_min;
                int  kernel_shape_w_max;
                float pool;
                int  data_cnt;

                kernel_shape_h_min = -pad_h_begin;
                kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                kernel_shape_w_min = -pad_w_begin;
                kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0.0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = 0.0;
                                    data_cnt = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) {
                                            if (count_include_pad != 0) {
                                                data_cnt += kernel_shape_w;
                                            }
                                            continue;
                                        }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { 
                                                if (count_include_pad != 0) {
                                                    data_cnt++;
                                                }
                                            } else {
                                                pool += X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                data_cnt++;
                                            }
                                        }
                                    }
                                    if (data_cnt > 0) {
                                        Y[n][c][h][w] = pool / data_cnt;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = 0.0;
                                    data_cnt = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) {
                                            if (count_include_pad != 0) {
                                                data_cnt++;
                                            }
                                            continue;
                                        }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) {
                                                if (count_include_pad != 0) {
                                                    data_cnt++;
                                                }
                                            } else {
                                                pool += X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                data_cnt++;
                                            }
                                        }
                                    }
                                    if (data_cnt > 0) {
                                        Y[n][c][h][w] = pool / data_cnt;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

