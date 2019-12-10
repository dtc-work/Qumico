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
__attribute__ ((aligned(SINGLE_ALIGN_SIZE))) int work_pad_int[65536];

#define mat_idx4(a, a_max, b, b_max, c, c_max, d, d_max) ((a)*(b_max)*(c_max)*(d_max) +(b)*(c_max)*(d_max) +(c)*(d_max) +(d))
#define mat_idx5(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max) +(b)*(c_max)*(d_max)*(e_max) +(c)*(d_max)*(e_max) +(d)*(e_max) +(e))
#define mat_idx6(a, a_max, b, b_max, c, c_max, d, d_max, e, e_max, f, f_max) ((a)*(b_max)*(c_max)*(d_max)*(e_max)*(f_max) +(b)*(c_max)*(d_max)*(e_max)*(f_max) +(c)*(d_max)*(e_max)*(f_max) +(d)*(e_max)*(f_max) +(e)*(f_max) +(f))
#define qlinearconv_CLAMP(x, low, high) ((x) > (high) ? (high) : ((x) < (low) ? (low) : (x)))




#undef TRANSPOSE
            void OpQLinearConv1(void *op_param, uint8_t X[1][3][128][128], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][3][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][64][64], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 3;
            const int  X_d = 1;
            const int  X_h = 128;
            const int  X_w = 128;
            const int  aligned_X_c = 3;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 128+0+1;
            const int  padded_X_w = 128+0+1;
//            const float _X_scale = 0.0078119998797774315;
//            const int  X_zero_point = 128;
            const int  W_m = 8;
            const int  W_c = 3;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 64;
            const int  Y_w = 64;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 64+0+1;
            const int  padded_Y_w = 64+0+1;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv2(void *op_param, uint8_t X[1][8][64][64], float X_scale[], uint8_t X_zero_point[], uint8_t W[8][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[8], uint8_t Y[1][8][64][64], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 64;
            const int  X_w = 64;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 64+1+1;
            const int  padded_X_w = 64+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 8;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 8;
            const int  Y_d = 1;
            const int  Y_h = 64;
            const int  Y_w = 64;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 8;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 64+1+1;
            const int  padded_Y_w = 64+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 8;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv3(void *op_param, uint8_t X[1][8][64][64], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][8][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][64][64], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 8;
            const int  X_d = 1;
            const int  X_h = 64;
            const int  X_w = 64;
            const int  aligned_X_c = 8;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 64+0+0;
            const int  padded_X_w = 64+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 16;
            const int  W_c = 8;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 64;
            const int  Y_w = 64;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 64+0+0;
            const int  padded_Y_w = 64+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv4(void *op_param, uint8_t X[1][16][64][64], float X_scale[], uint8_t X_zero_point[], uint8_t W[16][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[16], uint8_t Y[1][16][32][32], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 64;
            const int  X_w = 64;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 64+0+1;
            const int  padded_X_w = 64+0+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 16;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 16;
            const int  Y_d = 1;
            const int  Y_h = 32;
            const int  Y_w = 32;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 16;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 32+0+1;
            const int  padded_Y_w = 32+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 16;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv5(void *op_param, uint8_t X[1][16][32][32], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][16][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][32][32], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 16;
            const int  X_d = 1;
            const int  X_h = 32;
            const int  X_w = 32;
            const int  aligned_X_c = 16;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 32+0+0;
            const int  padded_X_w = 32+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 32;
            const int  W_c = 16;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 32;
            const int  Y_w = 32;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 32+0+0;
            const int  padded_Y_w = 32+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv6(void *op_param, uint8_t X[1][32][32][32], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][32][32], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 32;
            const int  X_w = 32;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 32+1+1;
            const int  padded_X_w = 32+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 32;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 32;
            const int  Y_w = 32;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 32+1+1;
            const int  padded_Y_w = 32+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 32;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv7(void *op_param, uint8_t X[1][32][32][32], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][32][32], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 32;
            const int  X_w = 32;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 32+0+0;
            const int  padded_X_w = 32+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 32;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 32;
            const int  Y_w = 32;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 32+0+0;
            const int  padded_Y_w = 32+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv8(void *op_param, uint8_t X[1][32][32][32], float X_scale[], uint8_t X_zero_point[], uint8_t W[32][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[32], uint8_t Y[1][32][16][16], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 32;
            const int  X_w = 32;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 32+0+1;
            const int  padded_X_w = 32+0+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 32;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 32;
            const int  Y_d = 1;
            const int  Y_h = 16;
            const int  Y_w = 16;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 32;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 16+0+1;
            const int  padded_Y_w = 16+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 32;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv9(void *op_param, uint8_t X[1][32][16][16], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][32][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][16][16], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 32;
            const int  X_d = 1;
            const int  X_h = 16;
            const int  X_w = 16;
            const int  aligned_X_c = 32;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 16+0+0;
            const int  padded_X_w = 16+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 64;
            const int  W_c = 32;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 16;
            const int  Y_w = 16;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 16+0+0;
            const int  padded_Y_w = 16+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv10(void *op_param, uint8_t X[1][64][16][16], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][16][16], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 16;
            const int  X_w = 16;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 16+1+1;
            const int  padded_X_w = 16+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 64;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 16;
            const int  Y_w = 16;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 16+1+1;
            const int  padded_Y_w = 16+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 64;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv11(void *op_param, uint8_t X[1][64][16][16], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][16][16], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 16;
            const int  X_w = 16;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 16+0+0;
            const int  padded_X_w = 16+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 64;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 16;
            const int  Y_w = 16;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 16+0+0;
            const int  padded_Y_w = 16+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv12(void *op_param, uint8_t X[1][64][16][16], float X_scale[], uint8_t X_zero_point[], uint8_t W[64][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[64], uint8_t Y[1][64][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 16;
            const int  X_w = 16;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 16+0+1;
            const int  padded_X_w = 16+0+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 64;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 64;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 64;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+1;
            const int  padded_Y_w = 8+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 64;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv13(void *op_param, uint8_t X[1][64][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][64][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 64;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 64;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 64;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv14(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+1+1;
            const int  padded_X_w = 8+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+1+1;
            const int  padded_Y_w = 8+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv15(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv16(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+1+1;
            const int  padded_X_w = 8+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+1+1;
            const int  padded_Y_w = 8+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv17(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv18(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+1+1;
            const int  padded_X_w = 8+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+1+1;
            const int  padded_Y_w = 8+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv19(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv20(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+1+1;
            const int  padded_X_w = 8+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+1+1;
            const int  padded_Y_w = 8+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv21(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv22(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+1+1;
            const int  padded_X_w = 8+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+1+1;
            const int  padded_Y_w = 8+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv23(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][8][8], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+0;
            const int  padded_X_w = 8+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 8;
            const int  Y_w = 8;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 8+0+0;
            const int  padded_Y_w = 8+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv24(void *op_param, uint8_t X[1][128][8][8], float X_scale[], uint8_t X_zero_point[], uint8_t W[128][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[128], uint8_t Y[1][128][4][4], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 8;
            const int  X_w = 8;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 8+0+1;
            const int  padded_X_w = 8+0+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 128;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 128;
            const int  Y_d = 1;
            const int  Y_h = 4;
            const int  Y_w = 4;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 128;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 4+0+1;
            const int  padded_Y_w = 4+0+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 128;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv25(void *op_param, uint8_t X[1][128][4][4], float X_scale[], uint8_t X_zero_point[], uint8_t W[256][128][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[256], uint8_t Y[1][256][4][4], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 128;
            const int  X_d = 1;
            const int  X_h = 4;
            const int  X_w = 4;
            const int  aligned_X_c = 128;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 4+0+0;
            const int  padded_X_w = 4+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 256;
            const int  W_c = 128;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 256;
            const int  Y_d = 1;
            const int  Y_h = 4;
            const int  Y_w = 4;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 256;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 4+0+0;
            const int  padded_Y_w = 4+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv26(void *op_param, uint8_t X[1][256][4][4], float X_scale[], uint8_t X_zero_point[], uint8_t W[256][1][3][3], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[256], uint8_t Y[1][256][4][4], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 256;
            const int  X_d = 1;
            const int  X_h = 4;
            const int  X_w = 4;
            const int  aligned_X_c = 256;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 4+1+1;
            const int  padded_X_w = 4+1+1;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 256;
            const int  W_c = 1;
            const int  W_kd = 1;
            const int  W_kh = 3;
            const int  W_kw = 3;
            const int  Y_n = 1;
            const int  Y_c = 256;
            const int  Y_d = 1;
            const int  Y_h = 4;
            const int  Y_w = 4;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 256;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 4+1+1;
            const int  padded_Y_w = 4+1+1;
            const int  B_n = 1;
            const int  dilation_d = 1;
            const int  dilation_h = 1;
            const int  dilation_w = 1;
            const int  group = 256;
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
        
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
                    

                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
//                    for (n=0; n<Y_n; n++) {
#pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                            current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                            if (current_h<0 || current_h>=X_h) { continue; }
                                            for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                if (current_w<0 || current_w>=X_w) { continue; }
                                                work_pad_int[mat_idx4( n, Y_n, oc, Y_c, h, Y_h, w, Y_w )] += (X[n][oc][current_h][current_w] - X_zero_point[0])
                                                                * (W[oc][oc/group][kh][kw] - W_zero_point[0]);
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv27(void *op_param, uint8_t X[1][256][4][4], float X_scale[], uint8_t X_zero_point[], uint8_t W[256][256][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[256], uint8_t Y[1][256][4][4], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 256;
            const int  X_d = 1;
            const int  X_h = 4;
            const int  X_w = 4;
            const int  aligned_X_c = 256;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 4+0+0;
            const int  padded_X_w = 4+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 256;
            const int  W_c = 256;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 256;
            const int  Y_d = 1;
            const int  Y_h = 4;
            const int  Y_w = 4;
//            const float Y_scale = 0.023528000339865685;
//            const int  Y_zero_point = 0;
            const int  aligned_Y_c = 256;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 4+0+0;
            const int  padded_Y_w = 4+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}










#undef TRANSPOSE
            void OpQLinearConv28(void *op_param, uint8_t X[1][256][1][1], float X_scale[], uint8_t X_zero_point[], uint8_t W[1001][256][1][1], float W_scale[1], uint8_t W_zero_point[1], float Y_scale[], uint8_t Y_zero_point[], int B[1001], uint8_t Y[1][1001][1][1], void *inputs_params, void* outputs_params)
{
    
            uint8_t* _X_pt = &X[0][0][0][0];
            uint8_t* _W_pt = &W[0][0][0][0];
            uint8_t* _Y_pt = &Y[0][0][0][0];
            
            const int  X_n = 1;
            const int  X_c = 256;
            const int  X_d = 1;
            const int  X_h = 1;
            const int  X_w = 1;
            const int  aligned_X_c = 256;
            const int  padded_X_d = 1+0+0;
            const int  padded_X_h = 1+0+0;
            const int  padded_X_w = 1+0+0;
//            const float _X_scale = 0.023528000339865685;
//            const int  X_zero_point = 0;
            const int  W_m = 1001;
            const int  W_c = 256;
            const int  W_kd = 1;
            const int  W_kh = 1;
            const int  W_kw = 1;
            const int  Y_n = 1;
            const int  Y_c = 1001;
            const int  Y_d = 1;
            const int  Y_h = 1;
            const int  Y_w = 1;
//            const float Y_scale = 0.1308329999446869;
//            const int  Y_zero_point = 96;
            const int  aligned_Y_c = 1001;
            const int  padded_Y_d = 1+0+0;
            const int  padded_Y_h = 1+0+0;
            const int  padded_Y_w = 1+0+0;
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
        
    #if !1 // Bias is None.
                    memset( (void *)work_pad_int, 0, sizeof(int) * Y_n * Y_c * Y_d * Y_h * Y_w );
    #endif // B
                    for (n=0; n<Y_n; n++) {
    #if 1 // Bias has elements.
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] = B[oc];
                                    }
                                }
                            }
                        }
    #endif // Bias
                    
    #pragma omp parallel for
                        for (oc=0; oc<Y_c; oc++) {
                            for (ic=0; ic<X_c; ic++) {
                                for (d=0; d<Y_d; d++) {
                                    for (h=0; h<Y_h; h++) {
                                        for (w=0; w<Y_w; w++) {
                                            for (kd=kernel_shape_d_min; kd<kernel_shape_d_max; kd++) {
                                                current_d = d*stride_d+kd*dilation_d-pad_d_begin;
                                                if (current_d<0 || current_d>=X_d) { continue; }
                                                for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                                    current_h = h*stride_h+kh*dilation_h-pad_h_begin;
                                                    if (current_h<0 || current_h>=X_h) { continue; }
                                                    for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                                        current_w = w*stride_w+kw*dilation_w-pad_w_begin;
                                                        if (current_w<0 || current_w>=X_w) { continue; }
//                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (X[n][ic][current_h][current_w] - X_zero_point[0])
//                                                                        * (W[oc][ic][kh][kw] - W_zero_point[0]);
                                                        work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] += (*(_X_pt + mat_idx5(n, X_n, ic, X_c, current_d, X_d, current_h, X_h, current_w, X_w)) - X_zero_point[0])
                                                                        * (*(_W_pt + mat_idx5(oc, Y_c, ic, X_c, kd, kernel_shape_d, kh, kernel_shape_h, kw, kernel_shape_w)) - W_zero_point[0]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (d=0; d<Y_d; d++) {
                                for (h=0; h<Y_h; h++) {
                                    for (w=0; w<Y_w; w++) {
//                                        Y[n][oc][h][w] = (uint8_t)round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w )] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]);
                                        *(_Y_pt + mat_idx5(n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)) = qlinearconv_CLAMP(round(work_pad_int[mat_idx5( n, Y_n, oc, Y_c, d, Y_d, h, Y_h, w, Y_w)] * X_scale[0] * W_scale[0] / Y_scale[0] + Y_zero_point[0]),0,255);
                                    }
                                }
                            }
                        }
                    
                    }
                
}

