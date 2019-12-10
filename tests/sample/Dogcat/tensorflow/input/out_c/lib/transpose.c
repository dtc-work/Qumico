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
} TransposeOpParam;



void OpTranspose1(void *op_param, uint8_t vi_image[1][224][224][3], uint8_t vi_image_Input0Transpose[1][3][224][224], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<224;k++){
                    for(int l=0;l<224;l++){
                        vi_image_Input0Transpose[0][j][k][l] = vi_image[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose2(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold_Output0Transpose[1][16][112][112], uint8_t vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold[1][112][112][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<112;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose3(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold[1][112][112][16], uint8_t vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold_Input0Transpose[1][16][112][112], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<112;l++){
                        vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_stem_conv_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose4(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu_Output0Transpose[1][16][112][112], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu[1][112][112][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<112;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose5(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu[1][112][112][16], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu_Input0Transpose[1][16][112][112], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<112;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose6(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold_Output0Transpose[1][8][112][112], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold[1][112][112][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<112;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose7(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold[1][112][112][8], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold_Input0Transpose[1][8][112][112], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<112;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_0_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose8(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_Output0Transpose[1][24][112][112], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu[1][112][112][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<112;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose9(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu[1][112][112][24], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_Input0Transpose[1][24][112][112], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<112;k++){
                    for(int l=0;l<112;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose10(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose11(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose12(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Output0Transpose[1][8][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose13(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[1][56][56][8], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Input0Transpose[1][8][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_1_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose14(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose15(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose16(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose17(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose18(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Output0Transpose[1][8][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose19(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output[1][56][56][8], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_2_output_Input0Transpose[1][8][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_2_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_2_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose20(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose21(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose22(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose23(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose24(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Output0Transpose[1][8][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold[1][56][56][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<8;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose25(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_output[1][56][56][8], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_3_output_Input0Transpose[1][8][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_3_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_3_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose26(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_Output0Transpose[1][24][56][56], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu[1][56][56][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<56;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose27(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu[1][56][56][24], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_Input0Transpose[1][24][56][56], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<56;k++){
                    for(int l=0;l<56;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose28(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1_Output0Transpose[1][24][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1[1][28][28][24], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<24;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose29(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1[1][28][28][24], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1_Input0Transpose[1][24][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<24;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose30(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Output0Transpose[1][16][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose31(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[1][28][28][16], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Input0Transpose[1][16][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_4_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose32(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_Output0Transpose[1][48][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu[1][28][28][48], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<48;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose33(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu[1][28][28][48], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_Input0Transpose[1][48][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<48;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose34(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1_Output0Transpose[1][48][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1[1][28][28][48], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<48;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose35(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1[1][28][28][48], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1_Input0Transpose[1][48][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<48;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose36(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Output0Transpose[1][16][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose37(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output[1][28][28][16], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_5_output_Input0Transpose[1][16][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_5_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_5_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose38(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_Output0Transpose[1][48][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu[1][28][28][48], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<48;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose39(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu[1][28][28][48], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_Input0Transpose[1][48][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<48;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose40(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1_Output0Transpose[1][48][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1[1][28][28][48], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<48;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose41(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1[1][28][28][48], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1_Input0Transpose[1][48][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<48;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose42(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Output0Transpose[1][16][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold[1][28][28][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<16;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose43(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_output[1][28][28][16], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_6_output_Input0Transpose[1][16][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_6_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_6_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose44(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_Output0Transpose[1][96][28][28], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu[1][28][28][96], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<28;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<96;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose45(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu[1][28][28][96], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_Input0Transpose[1][96][28][28], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<96;j++){
            for(int k=0;k<28;k++){
                    for(int l=0;l<28;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose46(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1_Output0Transpose[1][96][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1[1][14][14][96], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<96;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose47(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1[1][14][14][96], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1_Input0Transpose[1][96][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<96;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose48(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Output0Transpose[1][32][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose49(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[1][14][14][32], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Input0Transpose[1][32][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_7_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose50(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose51(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose52(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose53(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose54(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Output0Transpose[1][32][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose55(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output[1][14][14][32], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_8_output_Input0Transpose[1][32][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_8_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_8_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose56(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose57(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose58(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose59(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose60(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Output0Transpose[1][32][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose61(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output[1][14][14][32], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_9_output_Input0Transpose[1][32][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_9_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_9_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose62(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose63(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose64(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose65(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose66(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Output0Transpose[1][32][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose67(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[1][14][14][32], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Input0Transpose[1][32][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_10_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose68(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose69(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose70(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose71(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose72(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Output0Transpose[1][32][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold[1][14][14][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<32;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose73(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_output[1][14][14][32], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_11_output_Input0Transpose[1][32][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_11_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_11_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose74(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_Output0Transpose[1][192][14][14], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu[1][14][14][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<14;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose75(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu[1][14][14][192], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_Input0Transpose[1][192][14][14], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<14;k++){
                    for(int l=0;l<14;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose76(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1_Output0Transpose[1][192][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1[1][7][7][192], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<192;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose77(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1[1][7][7][192], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1_Input0Transpose[1][192][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<192;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose78(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Output0Transpose[1][64][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose79(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[1][7][7][64], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Input0Transpose[1][64][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_12_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose80(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose81(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose82(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose83(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose84(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Output0Transpose[1][64][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose85(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output[1][7][7][64], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_13_output_Input0Transpose[1][64][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_13_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_13_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose86(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose87(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose88(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose89(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose90(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Output0Transpose[1][64][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose91(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output[1][7][7][64], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_14_output_Input0Transpose[1][64][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_14_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_14_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose92(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose93(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose94(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose95(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose96(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Output0Transpose[1][64][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold[1][7][7][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<64;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose97(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_output[1][7][7][64], uint8_t vi_mnas_v4_a_035_1_feature_network_cell_15_output_Input0Transpose[1][64][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_cell_15_output_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_cell_15_output[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose98(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose99(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose100(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1_Output0Transpose[1][384][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1[1][7][7][384], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<384;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose101(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1[1][7][7][384], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1_Input0Transpose[1][384][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<384;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_Relu_1[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose102(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold_Output0Transpose[1][112][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold[1][7][7][112], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<112;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose103(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold[1][7][7][112], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold_Input0Transpose[1][112][7][7], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<112;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<7;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold_Input0Transpose[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_16_op_0_project_0_add_fold[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose104(void *op_param, uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_Output0Transpose[1][1280][7][7], uint8_t vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu[1][7][7][1280], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<7;j++){
            for(int k=0;k<7;k++){
                    for(int l=0;l<1280;l++){
                        vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu[0][j][k][l] = vi_mnas_v4_a_035_1_feature_network_lead_cell_17_op_0_Relu_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose105(void *op_param, uint8_t vi_mnas_v4_a_035_1_output_fc_weights_quant_FakeQuantWithMinMaxVars_transpose[2][1280], uint8_t vi_mnas_v4_a_035_1_output_fc_weights_quant_FakeQuantWithMinMaxVars_transpose_Input1Transpose[1280][2], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1280;i++){
            for(int j=0;j<2;j++){
                vi_mnas_v4_a_035_1_output_fc_weights_quant_FakeQuantWithMinMaxVars_transpose_Input1Transpose[i][j] = vi_mnas_v4_a_035_1_output_fc_weights_quant_FakeQuantWithMinMaxVars_transpose[j][i];
            }
    }

}

