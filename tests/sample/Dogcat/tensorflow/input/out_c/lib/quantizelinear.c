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
} QuantizeLinearOpParam;

#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))



void OpQuantizeLinear1(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_2_output_Quantize[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output[1][56][56][8], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][56][56][8];
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_2_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_2_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_2_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear2(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_3_output_Quantize[1][56][56][8] , float vi_mnas_v4_a_035_1_feature_network_cell_3_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_output[1][56][56][8], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][56][56][8];
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_3_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_3_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_3_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_3_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear3(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_5_output_Quantize[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output[1][28][28][16], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][28][28][16];
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_5_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_5_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_5_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear4(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_6_output_Quantize[1][28][28][16] , float vi_mnas_v4_a_035_1_feature_network_cell_6_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_output[1][28][28][16], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][28][28][16];
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_6_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_6_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_6_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_6_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear5(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_8_output_Quantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output[1][14][14][32], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][14][14][32];
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_8_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_8_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_8_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear6(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_9_output_Quantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output[1][14][14][32], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][14][14][32];
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_9_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_9_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_9_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear7(void *op_param,float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Quantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[1][14][14][32], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][14][14][32];
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear8(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_11_output_Quantize[1][14][14][32] , float vi_mnas_v4_a_035_1_feature_network_cell_11_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_output[1][14][14][32], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][14][14][32];
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_11_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_11_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_11_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_11_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear9(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_13_output_Quantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output[1][7][7][64], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][7][7][64];
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_13_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_13_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_13_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear10(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_14_output_Quantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output[1][7][7][64], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][7][7][64];
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_14_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_14_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_14_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear11(void *op_param,float vi_mnas_v4_a_035_1_feature_network_cell_15_output_Quantize[1][7][7][64] , float vi_mnas_v4_a_035_1_feature_network_cell_15_output_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_output_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_output[1][7][7][64], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][7][7][64];
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        temp_arr[i][j][k][l] = ROUND(vi_mnas_v4_a_035_1_feature_network_cell_15_output_Quantize[0][j][k][l] / vi_mnas_v4_a_035_1_feature_network_cell_15_output_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_cell_15_output_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_cell_15_output[i][j][k][l] = CLAMP(temp_arr[i][j][k][l], 0, 255);

                    }
            }
        }
    }

}









void OpQuantizeLinear12(void *op_param,float vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_Quantize[1][1280] , float vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_x_scale[1], uint8_t vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_x_zero_point[1], uint8_t vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean[1][1280], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][1280];
    for(int i=0;i<1;i++){
            for(int j=0;j<1280;j++){
                temp_arr[i][j] = ROUND(vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_Quantize[0][j] / vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_x_scale[0]) + vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean_x_zero_point[0];
            vi_mnas_v4_a_035_1_feature_network_feature_extractor_Mean[i][j] = CLAMP(temp_arr[i][j], 0, 255);

            }
    }

}









void OpQuantizeLinear13(void *op_param,float vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Add[1][2] , float vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_scale[1], uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_zero_point[1], uint8_t vi_mnas_v4_a_035_1_output_fc_BiasAdd[1][2], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][2];
    for(int i=0;i<1;i++){
            for(int j=0;j<2;j++){
                temp_arr[i][j] = ROUND(vi_mnas_v4_a_035_1_output_fc_BiasAdd_Output0Add[0][j] / vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_scale[0]) + vi_mnas_v4_a_035_1_output_fc_BiasAdd_y_zero_point[0];
            vi_mnas_v4_a_035_1_output_fc_BiasAdd[i][j] = CLAMP(temp_arr[i][j], 0, 255);

            }
    }

}









void OpQuantizeLinear14(void *op_param,float vi_scores_Quantize[1][2] , float vi_scores_x_scale[1], uint8_t vi_scores_x_zero_point[1], uint8_t vi_scores[1][2], void *inputs_params, void* outputs_params)
{      
        long temp_arr[1][2];
    for(int i=0;i<1;i++){
            for(int j=0;j<2;j++){
                temp_arr[i][j] = ROUND(vi_scores_Quantize[0][j] / vi_scores_x_scale[0]) + vi_scores_x_zero_point[0];
            vi_scores[i][j] = CLAMP(temp_arr[i][j], 0, 255);

            }
    }

}

