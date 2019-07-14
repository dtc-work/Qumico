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
    //----
	QMC_DTYPE_NONE // 8
} QMC_DTYPE;

