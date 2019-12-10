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



typedef struct {
    char* name;
} ReshapeOpParam;



void OpReshape1(void *op_param, uint8_t data[1][1][1][1001], long int shape[], uint8_t reshaped[1][1001], void *inputs_params, void* outputs_params) {
    
            uint8_t *_data = (uint8_t *)data;
            uint8_t *_reshaped = (uint8_t *)reshaped;

            int     data_elements = 1001;
            int     shape_elements = 1001;
            int     i;

            if (data_elements >= shape_elements) {
                for (i=0; i<shape_elements; i++) {
                    *(_reshaped +i) = *(_data +i);
                }
            } else {
                for (i=0; i<data_elements; i++) {
                    *(_reshaped +i) = *(_data +i);
                }
                for (; i<shape_elements; i++) {
                    *(_reshaped +i) = (uint8_t)0.0;
                }
            }
        
}

