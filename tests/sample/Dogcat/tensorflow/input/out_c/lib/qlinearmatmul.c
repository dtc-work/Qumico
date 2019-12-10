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
} QLinearMatMulOpParam;

#define qlinearmatmul_CLAMP(x, low, high) ((x) > (high) ? (high) : ((x) < (low) ? (low) : (x)))

int qlinearmatmul_ROUND(float x) {
    if (fabsf(x - (int)x) != 0.5) return (x >= 0.0 ? (int)(x + 0.5) : (int)(x - 0.5));
    else return (x >= 0.0 ? (int)((x + 0.5) - (int)(x + 0.5) % 2) : (int)((x - 0.5) - (int)(x - 0.5) % 2)); 
}



void OpQLinearMatMul1(void *op_param, uint8_t A[1][1280], float a_scale[1], uint8_t a_zero_point[1], uint8_t B[1280][2], float b_scale[1], uint8_t b_zero_point[1], float y_scale[1], uint8_t y_zero_point[1], uint8_t Y[1][2], void *inputs_params, void* outputs_params)
{
    
            const int   A_h = 1;
            const int   A_i = 1;
            const int   A_j = 1;
            const int   A_m = 1;
            const int   A_k = 1280;
            const int   B_h = 1;
            const int   B_i = 1;
            const int   B_j = 1;
            const int   B_k = 1280;
            const int   B_n = 2;
            const int   Y_h = 1;
            const int   Y_i = 1;
            const int   Y_j = 1;
            const int   Y_m = 1;
            const int   Y_n = 2;

            const int   A_h_o = 0;
            const int   A_i_o = 0;
            const int   A_j_o = 0;
            const int   B_h_o = 0;
            const int   B_i_o = 0;
            const int   B_j_o = 0;

            uint8_t *_A = (uint8_t *)A;
            uint8_t *_B = (uint8_t *)B;
            uint8_t *_Y = (uint8_t *)Y;
            int tmpA, tmpB, tmpY;
            uint8_t BT [1][1][1][1280][2];
            uint8_t *_BT = (uint8_t *)BT;

            uint8_t a_zero_point_mod[1];
            uint8_t b_zero_point_mod[2];
            uint8_t y_zero_point_mod[1];

            float a_scale_mod[1];
            float b_scale_mod[2];
            float y_scale_mod[1];
            float multiplier;

            int   h, i, j;
            int   ah, ai, aj;
            int   bh, bi, bj;
            int   k;
            int   m;
            int   n;

            int   tmpA_pos_h, tmpA_pos_i, tmpA_pos;
            int   tmpB_pos_h, tmpB_pos_i, tmpB_pos;
            int   tmpY_pos_h, tmpY_pos_i, tmpY_pos;

            memset( Y, (uint8_t)0, sizeof(*_Y)*Y_h*Y_i*Y_j*Y_m*Y_n );
        
#pragma omp parallel for
            for (m=0; m < A_m; m++) {
                a_zero_point_mod[m] = a_zero_point[0];
                a_scale_mod[m] = a_scale[0];
            }
            
#pragma omp parallel for
            for (n=0; n < B_n; n++) {
                b_zero_point_mod[n] = b_zero_point[0];
                b_scale_mod[n] = b_scale[0];
            }
            
#pragma omp parallel for
            for (m=0; m < A_m; m++) {
                y_zero_point_mod[m] = y_zero_point[0];
                y_scale_mod[m] = y_scale[0];
            }
            
            for (h=0; h < B_h; h++) {
                bh = (B_h_o > 1) ? h : 0;
                tmpB_pos_h = bh*(B_i*B_j*B_k*B_n);
                for (i=0; i < B_i; i++) {
                    bi = (B_i_o > 1) ? i : 0;
                    tmpB_pos_i = tmpB_pos_h + bi*(B_j*B_k*B_n);
                    for (j=0; j < B_j; j++) {
                        bj =  (B_j_o > 1) ? j : 0;
                        tmpB_pos = tmpB_pos_i + bj*(B_k*B_n);
#pragma omp parallel for private(n,k)
                        for (n=0; n < B_n; n++) {
                            for (k=0; k < B_k; k++) {
                                *(_BT + tmpB_pos + n*(B_k) + k) = *(_B + tmpB_pos + k*(B_n) + n);
                            }
                        }

                    }
                }
            }

            for (h=0; h < Y_h; h++) {
                ah = (A_h_o > 1) ? h : 0;
                bh = (B_h_o > 1) ? h : 0;
                tmpA_pos_h = ah*(A_i*A_j*A_m*A_k);
                tmpB_pos_h = bh*(B_i*B_j*B_k*B_n);
                tmpY_pos_h =  h*(Y_i*Y_j*Y_m*Y_n);
                for (i=0; i < Y_i; i++) {
                    ai = (A_i_o > 1) ? i : 0;
                    bi = (B_i_o > 1) ? i : 0;
                    tmpA_pos_i = tmpA_pos_h + ai*(A_j*A_m*A_k);
                    tmpB_pos_i = tmpB_pos_h + bi*(B_j*B_k*B_n);
                    tmpY_pos_i = tmpY_pos_h +  i*(Y_j*Y_m*Y_n);
                    for (j=0; j < Y_j; j++) {
                        aj =  (A_j_o > 1) ? j : 0;
                        bj =  (B_j_o > 1) ? j : 0;
                        tmpA_pos = tmpA_pos_i + aj*(A_m*A_k);
                        tmpB_pos = tmpB_pos_i + bj*(B_k*B_n);
                        tmpY_pos = tmpY_pos_i +  j*(Y_m*Y_n);
#pragma omp parallel for private(m,n,k,multiplier,tmpA,tmpB) reduction(+:tmpY)
                        for (m=0; m < Y_m; m++) {
                            for (n=0; n < Y_n; n++) {
                                tmpY = 0;
                                multiplier = a_scale_mod[m] * b_scale_mod[n] / y_scale_mod[m];
                                for (k=0; k < B_k; k++) {
                                    tmpA = *(_A  + tmpA_pos + m*(A_k) + k) - a_zero_point_mod[m];
                                    tmpB = *(_BT + tmpB_pos + n*(B_k) + k) - b_zero_point_mod[n];
                                    tmpY += tmpA * tmpB;
                                }
                                *(_Y + tmpY_pos + m*(Y_n) + n) = qlinearmatmul_CLAMP(qlinearmatmul_ROUND(multiplier * tmpY + y_zero_point_mod[m]),0,255);
                            }
                        }
                    }
                }
            }
        
}

