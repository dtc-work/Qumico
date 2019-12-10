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

#include "stdio.h"

typedef struct {
    char* name;
    int ndim;
    int* shape;
    void *value;
} QuantizeLinearOpParam;

#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))



void OpQuantizeLinear1(void *op_param,float vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Quantize[1][256][1][1] , float vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_x_scale[1], uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_x_zero_point[1], uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Transpose[1][256][1][1], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][256][1][1];
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<1;k++){
                    for(int l=0;l<1;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Quantize[0][j][0][0] / vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_x_scale[0]) + vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_x_zero_point[0];
            vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Transpose[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear2(void *op_param,float vi_MobilenetV1_Predictions_Reshape_1_Quantize[1][1001] , float vi_MobilenetV1_Predictions_Reshape_1_x_scale[1], uint8_t vi_MobilenetV1_Predictions_Reshape_1_x_zero_point[1], uint8_t vi_MobilenetV1_Predictions_Reshape_1[1][1001], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][1001];
    for(int i=0;i<1;i++){
            for(int j=0;j<1001;j++){
                temp_arr[i][j] = ROUND(vi_MobilenetV1_Predictions_Reshape_1_Quantize[0][j] / vi_MobilenetV1_Predictions_Reshape_1_x_scale[0]) + vi_MobilenetV1_Predictions_Reshape_1_x_zero_point[0];
            vi_MobilenetV1_Predictions_Reshape_1[i][j] = CLAMP(temp_arr[i][j], 0, 255);

            }
    }

}

