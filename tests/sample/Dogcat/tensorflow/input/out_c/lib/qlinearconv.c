/**
 * Copyright (c) 2019 Pasona Tech Inc. http://pasona.tech
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
 * Date Created : 2019-12-06 17:02:16
 * Version      : dev201910
 */

#include "math.h"

typedef struct {
    char* name;
    int   dilations[4];
    int   group;
    int   kernel_shape[2];
    int   pads[4];
    int   strides[2];
} QlinearConvOpParam;

#ifndef SINGLE_ALIGN_SIZE
#define SINGLE_ALIGN_SIZE (sizeof(int))
#endif
__attribute__ ((aligned(SINGLE_ALIGN_SIZE))) int work_pad_int[301056];

#define mat_idx4(a, a_max, b, b_max, c, c_max, d, d_max) ((a)*(b_max)*(c_max)*(d_max) +(b)*(c_max)*(d_max) +(c)*(d_max) +(d))
#define mat_idx5(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max) +(b)*(c_max)*(d_max)*(e_max) +(c)*(d_max)*(e_max) +(d)*(e_max) +(e))
#define mat_idx6(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max, f, f_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max)*(f_max) +(b)*(c_max)*(d_max)*(e_max)*(f_max) +(c)*(d_max)*(e_max)*(f_max) +(d)*(e_max)*(f_max) +(e)*(f_max) +(f))
#define qlinearconv_CLAMP(x, low, high) ((x) > (high) ? (high) : ((x) < (low) ? (low) : (x)))




#undef TRANSPOSE
            void OpQLinearConv1(void *op_param, uint8_t X[1][3][224][224], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][3][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][112][112], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 3;
            const int  X_d = 1;
            const int  X_h = 224;
            const int  X_w = 224;
            const int  aligned_X_c = 3;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 224+0+1;
            const int  padded_X_w = 224+0+1;
            const int  W_m = 16;
            const int  W_c = 3;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 112;
            const int  Y_w = 112;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 112+0+1;
            const int  padded_Y_w = 112+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 2;
            const int  stride_w = 2;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv2(void *op_param, uint8_t X[1][16][112][112], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][112][112], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 112;
            const int  X_w = 112;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 112+1+1;
            const int  padded_X_w = 112+1+1;
            const int  W_m = 16;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 112;
            const int  Y_w = 112;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 112+1+1;
            const int  padded_Y_w = 112+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 16;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv3(void *op_param, uint8_t X[1][16][112][112], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][16][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][112][112], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 112;
            const int  X_w = 112;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 112+0+0;
            const int  padded_X_w = 112+0+0;
            const int  W_m = 8;
            const int  W_c = 16;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 112;
            const int  Y_w = 112;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 112+0+0;
            const int  padded_Y_w = 112+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv4(void *op_param, uint8_t X[1][8][112][112], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][8][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][112][112], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 112;
            const int  X_w = 112;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 112+0+0;
            const int  padded_X_w = 112+0+0;
            const int  W_m = 24;
            const int  W_c = 8;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 112;
            const int  Y_w = 112;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 112+0+0;
            const int  padded_Y_w = 112+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv5(void *op_param, uint8_t X[1][24][112][112], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 112;
            const int  X_w = 112;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 112+0+1;
            const int  padded_X_w = 112+0+1;
            const int  W_m = 24;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+1;
            const int  padded_Y_w = 56+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 24;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 2;
            const int  stride_w = 2;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv6(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][24][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 8;
            const int  W_c = 24;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv7(void *op_param, uint8_t X[1][8][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][8][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 24;
            const int  W_c = 8;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv8(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+1+1;
            const int  padded_X_w = 56+1+1;
            const int  W_m = 24;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+1+1;
            const int  padded_Y_w = 56+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 24;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv9(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][24][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 8;
            const int  W_c = 24;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv10(void *op_param, uint8_t X[1][8][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][8][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 24;
            const int  W_c = 8;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv11(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+1+1;
            const int  padded_X_w = 56+1+1;
            const int  W_m = 24;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+1+1;
            const int  padded_Y_w = 56+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 24;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv12(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][24][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 8;
            const int  W_c = 24;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv13(void *op_param, uint8_t X[1][8][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][8][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][56][56], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+0+0;
            const int  padded_X_w = 56+0+0;
            const int  W_m = 24;
            const int  W_c = 8;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 56;
            const int  Y_w = 56;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 56+0+0;
            const int  padded_Y_w = 56+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv14(void *op_param, uint8_t X[1][24][56][56], float X_scale[], uint8_t X_zero_point[], uint8_t W[24][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[24], uint8_t Y[1][24][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 56;
            const int  X_w = 56;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 56+1+2;
            const int  padded_X_w = 56+1+2;
            const int  W_m = 24;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 24;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 24;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+1+2;
            const int  padded_Y_w = 28+1+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 24;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 2;
            const int  stride_w = 2;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv15(void *op_param, uint8_t X[1][24][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][24][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 24;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 24;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 16;
            const int  W_c = 24;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv16(void *op_param, uint8_t X[1][16][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[48][16][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[48], uint8_t Y[1][48][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 48;
            const int  W_c = 16;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 48;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 48;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv17(void *op_param, uint8_t X[1][48][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[48][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[48], uint8_t Y[1][48][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 48;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 48;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+2+2;
            const int  padded_X_w = 28+2+2;
            const int  W_m = 48;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 48;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 48;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+2+2;
            const int  padded_Y_w = 28+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 48;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv18(void *op_param, uint8_t X[1][48][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][48][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 48;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 48;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 16;
            const int  W_c = 48;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv19(void *op_param, uint8_t X[1][16][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[48][16][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[48], uint8_t Y[1][48][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 48;
            const int  W_c = 16;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 48;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 48;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv20(void *op_param, uint8_t X[1][48][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[48][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[48], uint8_t Y[1][48][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 48;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 48;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+2+2;
            const int  padded_X_w = 28+2+2;
            const int  W_m = 48;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 48;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 48;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+2+2;
            const int  padded_Y_w = 28+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 48;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv21(void *op_param, uint8_t X[1][48][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][48][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 48;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 48;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 16;
            const int  W_c = 48;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv22(void *op_param, uint8_t X[1][16][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[96][16][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[96], uint8_t Y[1][96][28][28], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+0+0;
            const int  padded_X_w = 28+0+0;
            const int  W_m = 96;
            const int  W_c = 16;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 96;
            const int  Y_d = 1;
            const int  Y_h = 28;
            const int  Y_w = 28;
            const int  aligned_Y_c = 96;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 28+0+0;
            const int  padded_Y_w = 28+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv23(void *op_param, uint8_t X[1][96][28][28], float X_scale[], uint8_t X_zero_point[], uint8_t W[96][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[96], uint8_t Y[1][96][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 96;
            const int  X_d = 1;
            const int  X_h = 28;
            const int  X_w = 28;
            const int  aligned_X_c = 96;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 28+1+2;
            const int  padded_X_w = 28+1+2;
            const int  W_m = 96;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 96;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 96;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+1+2;
            const int  padded_Y_w = 14+1+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 96;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 2;
            const int  stride_w = 2;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv24(void *op_param, uint8_t X[1][96][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][96][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 96;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 96;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 32;
            const int  W_c = 96;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv25(void *op_param, uint8_t X[1][32][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 192;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv26(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+2+2;
            const int  padded_X_w = 14+2+2;
            const int  W_m = 192;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+2+2;
            const int  padded_Y_w = 14+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 192;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv27(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][192][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 32;
            const int  W_c = 192;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv28(void *op_param, uint8_t X[1][32][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 192;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv29(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+2+2;
            const int  padded_X_w = 14+2+2;
            const int  W_m = 192;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+2+2;
            const int  padded_Y_w = 14+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 192;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv30(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][192][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 32;
            const int  W_c = 192;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv31(void *op_param, uint8_t X[1][32][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 192;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv32(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+1+1;
            const int  padded_X_w = 14+1+1;
            const int  W_m = 192;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+1+1;
            const int  padded_Y_w = 14+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 192;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv33(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][192][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 32;
            const int  W_c = 192;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv34(void *op_param, uint8_t X[1][32][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 192;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv35(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+1+1;
            const int  padded_X_w = 14+1+1;
            const int  W_m = 192;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+1+1;
            const int  padded_Y_w = 14+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 192;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv36(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][192][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 32;
            const int  W_c = 192;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv37(void *op_param, uint8_t X[1][32][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][14][14], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+0+0;
            const int  padded_X_w = 14+0+0;
            const int  W_m = 192;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 14;
            const int  Y_w = 14;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 14+0+0;
            const int  padded_Y_w = 14+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv38(void *op_param, uint8_t X[1][192][14][14], float X_scale[], uint8_t X_zero_point[], uint8_t W[192][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[192], uint8_t Y[1][192][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 14;
            const int  X_w = 14;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 14+1+2;
            const int  padded_X_w = 14+1+2;
            const int  W_m = 192;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 192;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 192;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+1+2;
            const int  padded_Y_w = 7+1+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 192;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 2;
            const int  stride_w = 2;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv39(void *op_param, uint8_t X[1][192][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][192][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 192;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 192;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 64;
            const int  W_c = 192;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv40(void *op_param, uint8_t X[1][64][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 384;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv41(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+2+2;
            const int  padded_X_w = 7+2+2;
            const int  W_m = 384;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+2+2;
            const int  padded_Y_w = 7+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 384;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv42(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][384][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 64;
            const int  W_c = 384;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv43(void *op_param, uint8_t X[1][64][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 384;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv44(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+2+2;
            const int  padded_X_w = 7+2+2;
            const int  W_m = 384;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+2+2;
            const int  padded_Y_w = 7+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 384;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv45(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][384][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 64;
            const int  W_c = 384;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv46(void *op_param, uint8_t X[1][64][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 384;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv47(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][1][5][5], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+2+2;
            const int  padded_X_w = 7+2+2;
            const int  W_m = 384;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 5;
            const int  W_kw = 5;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+2+2;
            const int  padded_Y_w = 7+2+2;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 384;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 5;
            const int  kernel_shape_w = 5;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 2;
            const int  pad_w_begin = 2;
            const int  pad_d_end = 0;
            const int  pad_h_end = 2;
            const int  pad_w_end = 2;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 5;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 5;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv48(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][384][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 64;
            const int  W_c = 384;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv49(void *op_param, uint8_t X[1][64][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 384;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv50(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[384][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[384], uint8_t Y[1][384][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+1+1;
            const int  padded_X_w = 7+1+1;
            const int  W_m = 384;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 384;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 384;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+1+1;
            const int  padded_Y_w = 7+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 384;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 3;
            const int  kernel_shape_w = 3;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 1;
            const int  pad_w_begin = 1;
            const int  pad_d_end = 0;
            const int  pad_h_end = 1;
            const int  pad_w_end = 1;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 3;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 3;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
#pragma omp parallel for
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                                int w_zero_point_shift = 0;
                                int x_zero_point_shift = 0;
                                int kernel_counter = 0;
                                for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                    current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                    if (current_d<0 || current_d>=X_d) { continue; }
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                        if (current_h<0 || current_h>=X_h) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                            if (current_w<0 || current_w>=X_w) { continue; }
                                            work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, oc, Y_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                * (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d_max, kh, kernel_shape_h_max, kw, kernel_shape_w_max)));
                                            x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, oc, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                            w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, (ic/group), (X_c/group), kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                            kernel_counter++;
                                        }
                                    }
                                }
                    
                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                     - W_zero_point[0] * x_zero_point_shift
                                                                                                     + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv51(void *op_param, uint8_t X[1][384][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[112][384][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[112], uint8_t Y[1][112][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 384;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 384;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 112;
            const int  W_c = 384;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 112;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 112;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}










#undef TRANSPOSE
            void OpQLinearConv52(void *op_param, uint8_t X[1][112][7][7], float X_scale[], uint8_t X_zero_point[], uint8_t W[1280][112][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[1280], uint8_t Y[1][1280][7][7], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 112;
            const int  X_d = 1;
            const int  X_h = 7;
            const int  X_w = 7;
            const int  aligned_X_c = 112;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 7+0+0;
            const int  padded_X_w = 7+0+0;
            const int  W_m = 1280;
            const int  W_c = 112;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 1280;
            const int  Y_d = 1;
            const int  Y_h = 7;
            const int  Y_w = 7;
            const int  aligned_Y_c = 1280;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 7+0+0;
            const int  padded_Y_w = 7+0+0;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 1;
            const int  kernel_shape_d = 1;
            const int  kernel_shape_h = 1;
            const int  kernel_shape_w = 1;
            const int  pad_d_begin = 0;
            const int  pad_h_begin = 0;
            const int  pad_w_begin = 0;
            const int  pad_d_end = 0;
            const int  pad_h_end = 0;
            const int  pad_w_end = 0;
            const int  stride_d = 1;
            const int  stride_h = 1;
            const int  stride_w = 1;

            int  n;
            int  d, h, w;
            int  kd, kh, kw;
            int  ic, oc;
            int  oc1, oc2;
            int  current_d, current_h, current_w;

            const int  kernel_shape_d_min = 0;
            const int  kernel_shape_d_max = 1;
            const int  kernel_shape_h_min = 0;
            const int  kernel_shape_h_max = 1;
            const int  kernel_shape_w_min = 0;
            const int  kernel_shape_w_max = 1;
        
            for (n=0; n<Y_n; n++) {
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
//                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = B[oc];
                                work_pad_int[mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] = B[oc];
                            }
                        }
                    }
                }
                
#pragma omp parallel for
                for (oc=0; oc<Y_c; oc++) {
                    for (ic=0; ic<X_c; ic++) {
                        for (d=0; d<Y_d; d++) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    int w_zero_point_shift = 0;
                                    int x_zero_point_shift = 0;
                                    int kernel_counter = 0;
                                    for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                        current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                        if (current_d<0 || current_d>=X_d) { continue; }
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                    * (W[oc][ic][kh][kw] - W_zero_point[0]);
//                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
//                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)))
                                                                    * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                x_zero_point_shift += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)));
                                                w_zero_point_shift += (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)));
                                                kernel_counter++;
                                            }
                                        }
                                    }
                    
                                    work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += - X_zero_point[0] * w_zero_point_shift
                                                                                                        - W_zero_point[0] * x_zero_point_shift
                                                                                                        + X_zero_point[0] * W_zero_point[0] * kernel_counter;
                        
                                }
                            }
                        }
                    }
                    for (d=0; d<Y_d; d++) {
                        for (h=0; h<Y_h; h++) {
                            for (w=0; w<Y_w; w++) {
                    
//                                Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                        
                            }
                        }
                    }
                }
                    
            }
                
}

