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
} TransposeOpParam;



void OpTranspose1(void *op_param, uint8_t vi_input[1][128][128][3], uint8_t vi_input_Input0Transpose[1][3][128][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<128;k++){
                    for(int l=0;l<128;l++){
                        vi_input_Input0Transpose[0][j][k][l] = vi_input[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose2(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6_Output0Transpose[1][8][64][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6[1][64][64][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose3(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6[1][64][64][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6_Input0Transpose[1][8][64][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_0_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose4(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6_Output0Transpose[1][8][64][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6[1][64][64][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose5(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6[1][64][64][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6_Input0Transpose[1][8][64][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_1_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose6(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6_Output0Transpose[1][16][64][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6[1][64][64][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose7(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6[1][64][64][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6_Input0Transpose[1][16][64][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<64;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_1_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose8(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6_Output0Transpose[1][16][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6[1][32][32][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose9(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6[1][32][32][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6_Input0Transpose[1][16][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_2_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose10(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6_Output0Transpose[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose11(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6_Input0Transpose[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_2_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose12(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6_Output0Transpose[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose13(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6_Input0Transpose[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_3_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose14(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6_Output0Transpose[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose15(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6[1][32][32][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6_Input0Transpose[1][32][32][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<32;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_3_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose16(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6_Output0Transpose[1][32][16][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6[1][16][16][32], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<32;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose17(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6[1][16][16][32], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6_Input0Transpose[1][32][16][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<32;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_4_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose18(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6_Output0Transpose[1][64][16][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6[1][16][16][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose19(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6[1][16][16][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6_Input0Transpose[1][64][16][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_4_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose20(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6_Output0Transpose[1][64][16][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6[1][16][16][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose21(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6[1][16][16][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6_Input0Transpose[1][64][16][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_5_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose22(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6_Output0Transpose[1][64][16][16], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6[1][16][16][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<16;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose23(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6[1][16][16][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6_Input0Transpose[1][64][16][16], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<16;k++){
                    for(int l=0;l<16;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_5_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose24(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6_Output0Transpose[1][64][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6[1][8][8][64], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<64;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose25(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6[1][8][8][64], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6_Input0Transpose[1][64][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<64;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_6_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose26(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose27(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_6_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose28(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose29(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_7_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose30(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose31(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_7_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose32(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose33(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_8_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose34(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose35(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_8_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose36(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose37(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_9_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose38(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose39(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_9_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose40(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose41(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_10_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose42(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose43(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_10_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose44(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose45(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_11_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose46(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6_Output0Transpose[1][128][8][8], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6[1][8][8][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose47(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6[1][8][8][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6_Input0Transpose[1][128][8][8], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<8;k++){
                    for(int l=0;l<8;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_11_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose48(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6_Output0Transpose[1][128][4][4], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6[1][4][4][128], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<128;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose49(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6[1][4][4][128], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6_Input0Transpose[1][128][4][4], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<128;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<4;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_12_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose50(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6_Output0Transpose[1][256][4][4], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6[1][4][4][256], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<256;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose51(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6[1][4][4][256], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6_Input0Transpose[1][256][4][4], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<4;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_12_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose52(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6_Output0Transpose[1][256][4][4], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6[1][4][4][256], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<256;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose53(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6[1][4][4][256], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6_Input0Transpose[1][256][4][4], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<4;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6_Input0Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_13_depthwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose54(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Output0Transpose[1][256][4][4], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6[1][4][4][256], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<256;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Output0Transpose[0][l][j][k];
                    }
            }
        }
    }

}







void OpTranspose55(void *op_param, uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6[1][4][4][256], uint8_t vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Transpose[1][256][4][4], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<4;k++){
                    for(int l=0;l<4;l++){
                        vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6_Transpose[0][j][k][l] = vi_MobilenetV1_MobilenetV1_Conv2d_13_pointwise_Relu6[0][k][l][j];
                    }
            }
        }
    }

}







void OpTranspose56(void *op_param, uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Transpose[1][256][1][1], uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool[1][1][1][256], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<1;j++){
            for(int k=0;k<1;k++){
                    for(int l=0;l<256;l++){
                        vi_MobilenetV1_Logits_AvgPool_1a_AvgPool[0][0][0][l] = vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Transpose[0][l][0][0];
                    }
            }
        }
    }

}







void OpTranspose57(void *op_param, uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool[1][1][1][256], uint8_t vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Input0Transpose[1][256][1][1], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<256;j++){
            for(int k=0;k<1;k++){
                    for(int l=0;l<1;l++){
                        vi_MobilenetV1_Logits_AvgPool_1a_AvgPool_Input0Transpose[0][j][0][0] = vi_MobilenetV1_Logits_AvgPool_1a_AvgPool[0][0][0][j];
                    }
            }
        }
    }

}







void OpTranspose58(void *op_param, uint8_t vi_MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd_Output0Transpose[1][1001][1][1], uint8_t vi_MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd[1][1][1][1001], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++){
        for(int j=0;j<1;j++){
            for(int k=0;k<1;k++){
                    for(int l=0;l<1001;l++){
                        vi_MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd[0][0][0][l] = vi_MobilenetV1_Logits_Conv2d_1c_1x1_BiasAdd_Output0Transpose[0][l][0][0];
                    }
            }
        }
    }

}

