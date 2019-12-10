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

#pragma once
// Node
typedef struct Node{
    void * op_param;
    void * outputs;
    int output_ndim;
    int * output_shape;
} Node;

// Model
typedef struct Model{
    int node_cnt;
    Node* graph;
    size_t graph_size;
	void *graph_data;
} Model;



typedef enum {
    QMC_INT32, // 0
	QMC_FLOAT32, // 1
	QMC_QINT8, // 2
	QMC_UINT8, // 3
	QMC_INT8, // 4
	QMC_FIX8, // 5
	QMC_FIX16, // 6
	QMC_FLOAT16, // 7
    QMC_INT64, // 8
	//----
	QMC_DTYPE_NONE // 9
} QMC_DTYPE;

