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
} DequantizeLinearOpParam;



void OpDequantizeLinear1(void *op_param,uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Transpose[1][256][4][4] , float vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_x_scale[1], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_x_zero_point[1], float vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Dequantize[1][256][4][4], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<4;l++){
                        int x_temp = vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Transpose[0][j][k][l];
            int x_zero_temp = vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_x_zero_point[0];
            vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear2(void *op_param,uint8_t vi_MobilenetV1_Logits_SpatialSqueeze[1][1001] , float vi_MobilenetV1_Logits_SpatialSqueeze_x_scale[1], uint8_t vi_MobilenetV1_Logits_SpatialSqueeze_x_zero_point[1], float vi_MobilenetV1_Logits_SpatialSqueeze_Dequantize[1][1001], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
            for(int j=0;j<1001;j++){
                int x_temp = vi_MobilenetV1_Logits_SpatialSqueeze[0][j];
            int x_zero_temp = vi_MobilenetV1_Logits_SpatialSqueeze_x_zero_point[0];
            vi_MobilenetV1_Logits_SpatialSqueeze_Dequantize[i][j] = (x_temp - x_zero_temp) * vi_MobilenetV1_Logits_SpatialSqueeze_x_scale[0];
            }
    }

}

