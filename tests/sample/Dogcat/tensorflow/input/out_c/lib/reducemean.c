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



typedef struct {
    char* name;
    int ndim;
    int* shape;
    void *value;
} ReduceMeanOpParam;



void OpReduceMean1(void *op_param,float input[1][7][7][1280], float output[1][1280], void *inputs_params, void* outputs_params)
{
    for(int i=0;i<1;i++ ){
        for(int j=0;j<7;j++ ){
            for(int k=0;k<7;k++ ){
                for(int l=0;l<1280;l++ ){
                output[i][l]+=input[i][j][k][l];
                }
            }
        }
    }

    for(int i=0;i<1;i++ ){
        for(int j=0;j<1280;j++ ){
        output[i][j]/=49;
        }
    }
}

