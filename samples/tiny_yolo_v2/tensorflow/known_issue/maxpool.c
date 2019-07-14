#include "math.h"
#include "float.h"

typedef struct {
    char* name;
    int   kernel_shape[2];
    int   pads[4];
    int   storage_order;
    int   strides[4];
} MaxPoolOpParam;



void OpMaxPool1(void *op_param, float X[1][16][416][416], float Y[1][16][208][208], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 16;
                const int  X_h = 416;
                const int  X_w = 416;
                const int  Y_n = 1;
                const int  Y_c = 16;
                const int  Y_h = 208;
                const int  Y_w = 208;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 0;
                const int  stride_h = 2;
                const int  stride_w = 2;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

#include "math.h"
#include "float.h"





void OpMaxPool2(void *op_param, float X[1][32][208][208], float Y[1][32][104][104], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 32;
                const int  X_h = 208;
                const int  X_w = 208;
                const int  Y_n = 1;
                const int  Y_c = 32;
                const int  Y_h = 104;
                const int  Y_w = 104;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 0;
                const int  stride_h = 2;
                const int  stride_w = 2;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

#include "math.h"
#include "float.h"





void OpMaxPool3(void *op_param, float X[1][64][104][104], float Y[1][64][52][52], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 64;
                const int  X_h = 104;
                const int  X_w = 104;
                const int  Y_n = 1;
                const int  Y_c = 64;
                const int  Y_h = 52;
                const int  Y_w = 52;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 0;
                const int  stride_h = 2;
                const int  stride_w = 2;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

#include "math.h"
#include "float.h"





void OpMaxPool4(void *op_param, float X[1][128][52][52], float Y[1][128][26][26], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 128;
                const int  X_h = 52;
                const int  X_w = 52;
                const int  Y_n = 1;
                const int  Y_c = 128;
                const int  Y_h = 26;
                const int  Y_w = 26;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 0;
                const int  stride_h = 2;
                const int  stride_w = 2;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

#include "math.h"
#include "float.h"





void OpMaxPool5(void *op_param, float X[1][256][26][26], float Y[1][256][13][13], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 256;
                const int  X_h = 26;
                const int  X_w = 26;
                const int  Y_n = 1;
                const int  Y_c = 256;
                const int  Y_h = 13;
                const int  Y_w = 13;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 0;
                const int  stride_h = 2;
                const int  stride_w = 2;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

#include "math.h"
#include "float.h"





void OpMaxPool6(void *op_param, float X[1][512][13][13], float Y[1][512][13][13], void *inputs_params, void* outputs_params) {
    
                const int  X_n = 1;
                const int  X_c = 512;
                const int  X_h = 13;
                const int  X_w = 13;
                const int  Y_n = 1;
                const int  Y_c = 512;
                const int  Y_h = 13;
                const int  Y_w = 13;
//                const int  kernel_shape_h = 1;
                const int  kernel_shape_h = 2;
                const int  kernel_shape_w = 2;
                const int  pad_h_begin = 0;
                const int  pad_w_begin = 0;
                const int  pad_h_end = 0;
                const int  pad_w_end = 1;
                const int  stride_h = 1;
                const int  stride_w = 1;
                const int  storage_order = 0;

                int  n;
                int  c;
                int  h, w;
                int  kh, kw;
                float pool;
                int  max_flag;

                const int  kernel_shape_h_min = -pad_h_begin;
                const int  kernel_shape_h_max = (kernel_shape_h - pad_h_begin);
                const int  kernel_shape_w_min = -pad_w_begin;
                const int  kernel_shape_w_max = (kernel_shape_w - pad_w_begin);

                memset( (void *)Y, 0, sizeof(Y[0][0][0][0]) * Y_n * Y_c * Y_h * Y_w );

                for (n=0; n<Y_n; n++) {
                    for (c=0; c<Y_c; c++) {
                        if (storage_order == 0) {
                            for (h=0; h<Y_h; h++) {
                                for (w=0; w<Y_w; w++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        } else {
                            for (w=0; w<Y_w; w++) {
                                for (h=0; h<Y_h; h++) {
                                    pool = -DBL_MAX;
                                    max_flag = 0;
                                    for (kh=kernel_shape_h_min; kh<kernel_shape_h_max; kh++) {
                                        if ((h*stride_h+kh < 0) || (h*stride_h+kh >= X_h)) { continue; }
                                        for (kw=kernel_shape_w_min; kw<kernel_shape_w_max; kw++) {
                                            if ((w*stride_w+kw < 0) || (w*stride_w+kw >= X_w)) { continue; }
                                            if (pool < X[n][c][h*stride_h+kh][w*stride_w+kw]) {
                                                pool = X[n][c][h*stride_h+kh][w*stride_w+kw];
                                                max_flag = 1;
                                            }
                                        }
                                    }
                                    if (max_flag) {
                                        Y[n][c][h][w] = pool;
                                    }
                                }
                            }
                        }
                    }
                }
            
}

