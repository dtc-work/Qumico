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

#include "stdio.h"

typedef struct {
    char* name;
    int ndim;
    int* shape;
    void *value;
} DequantizeLinearOpParam;



void OpDequantizeLinear1(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Dequantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear2(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Dequantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear3(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_2_output_Dequantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_2_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_2_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear4(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Dequantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear5(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Dequantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear6(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Dequantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear7(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_5_output_Dequantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_5_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_5_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear8(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Dequantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear9(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear10(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear11(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_8_output_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_8_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_8_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear12(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear13(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_9_output_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_9_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_9_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear14(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear15(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear16(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Dequantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear17(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear18(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear19(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_13_output_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_13_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_13_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear20(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear21(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_14_output_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_14_output[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_14_output_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear22(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Dequantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear23(void *op_param,uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu[1][7][7][1280] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_x_zero_point[1], float vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_Dequantize[1][7][7][1280], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<1280;l++){
                        int x_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu[0][j][k][l];
            int x_zero_temp = vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_Dequantize[i][j][k][l] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_x_scale[0];
                    }
            }
        }
    }

}







void OpDequantizeLinear24(void *op_param,uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0QLinearMatmul[1][2] , float vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_scale[1], uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_zero_point[1], float vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Dequantize[1][2], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
            for(int j=0;j<2;j++){
                int x_temp = vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0QLinearMatmul[0][j];
            int x_zero_temp = vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_zero_point[0];
            vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Dequantize[i][j] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_scale[0];
            }
    }

}







void OpDequantizeLinear25(void *op_param,int vi_mnas_v4_a_035_1_output_fc_MatMul_bias[2] , float vi_mnas_v4_a_035_1_output_fc_MatMul_bias_w_scale[1], int vi_mnas_v4_a_035_1_output_fc_MatMul_bias_w_zero_point[1], float vi_mnas_v4_a_035_1_output_fc_MatMul_bias_Input2Cast[2], void *inputs_params, void* outputs_params)
{
        for(int i=0;i<2;i++){
            int x_temp = vi_mnas_v4_a_035_1_output_fc_MatMul_bias[i];
            int x_zero_temp = vi_mnas_v4_a_035_1_output_fc_MatMul_bias_w_zero_point[0];
            vi_mnas_v4_a_035_1_output_fc_MatMul_bias_Input2Cast[i] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_output_fc_MatMul_bias_w_scale[0];
        }

}







void OpDequantizeLinear26(void *op_param,uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd[1][2] , float vi_mnas_v4_a_035_1_output_fc_BiasAdd_x_scale[1], uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd_x_zero_point[1], float vi_mnas_v4_a_035_1_output_fc_BiasAdd_Dequantize[1][2], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
            for(int j=0;j<2;j++){
                int x_temp = vi_mnas_v4_a_035_1_output_fc_BiasAdd[0][j];
            int x_zero_temp = vi_mnas_v4_a_035_1_output_fc_BiasAdd_x_zero_point[0];
            vi_mnas_v4_a_035_1_output_fc_BiasAdd_Dequantize[i][j] = (x_temp - x_zero_temp) * vi_mnas_v4_a_035_1_output_fc_BiasAdd_x_scale[0];
            }
    }

}

