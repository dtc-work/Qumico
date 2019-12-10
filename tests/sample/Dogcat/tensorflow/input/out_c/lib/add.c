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
} AddOpParam;



void OpAdd1(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Dequantize[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Dequantize[1][56][56][8], float vi_mnas_v4_a_035_1_feature_network_cell_2_output_Quantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd2(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Dequantize[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_2_output_Dequantize[1][56][56][8], float vi_mnas_v4_a_035_1_feature_network_cell_3_output_Quantize[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_2_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd3(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Dequantize[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Dequantize[1][28][28][16], float vi_mnas_v4_a_035_1_feature_network_cell_5_output_Quantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd4(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Dequantize[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_5_output_Dequantize[1][28][28][16], float vi_mnas_v4_a_035_1_feature_network_cell_6_output_Quantize[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_5_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd5(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Dequantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Dequantize[1][14][14][32], float vi_mnas_v4_a_035_1_feature_network_cell_8_output_Quantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd6(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Dequantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_8_output_Dequantize[1][14][14][32], float vi_mnas_v4_a_035_1_feature_network_cell_9_output_Quantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_8_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd7(void *op_param,float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Dequantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_9_output_Dequantize[1][14][14][32], float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Quantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_9_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd8(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Dequantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Dequantize[1][14][14][32], float vi_mnas_v4_a_035_1_feature_network_cell_11_output_Quantize[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd9(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Dequantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Dequantize[1][7][7][64], float vi_mnas_v4_a_035_1_feature_network_cell_13_output_Quantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd10(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Dequantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_13_output_Dequantize[1][7][7][64], float vi_mnas_v4_a_035_1_feature_network_cell_14_output_Quantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_13_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd11(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Dequantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_14_output_Dequantize[1][7][7][64], float vi_mnas_v4_a_035_1_feature_network_cell_15_output_Quantize[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_output_Quantize[i][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Dequantize[0][j][k][l] + vi_mnas_v4_a_035_1_feature_network_cell_14_output_Dequantize[0][j][k][l];
                    }
            }
        }
    }

}







void OpAdd12(void *op_param,float vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Dequantize[1][2] , float vi_mnas_v4_a_035_1_output_fc_MatMul_bias_Input2Cast[2], float vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Add[1][2], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
            for(int j=0;j<2;j++){
                vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Add[i][j] = vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Dequantize[0][j] + vi_mnas_v4_a_035_1_output_fc_MatMul_bias_Input2Cast[j];
            }
    }

}

