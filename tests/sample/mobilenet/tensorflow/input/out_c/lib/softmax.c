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
} SoftmaxOpParam;



void OpSoftmax1(void *op_param, float input[1][1001], float output[1][1001], void *inputs_params, void* outputs_params)
{
    
            float   *_input = (float *)input;
            float   *_output = (float *)output;
            int    batch_size = 1;
            int    num = 1001;

            int    i;
            int    batch;
            float  max, sum;

            for (batch=0; batch<batch_size; batch++) {
                sum = 0.0;
                max = -HUGE_VAL;
                for (i=0; i<num; i++) {
                    if (*(_input + batch*num +i) > max) {
                        max = *(_input + batch*num +i);
                    }
                }
                for (i=0; i<num; i++) {
                    *(_output + batch*num +i) = expf(*(_input + batch*num +i) - max);
                    sum += *(_output + batch*num +i);
                }
                for (i=0; i<num; i++) {
                    *(_output + batch*num +i) /= sum;
                }
            }
        
}

