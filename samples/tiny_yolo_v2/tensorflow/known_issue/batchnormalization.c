#include <math.h>

typedef struct {
    char* name;
    double epsilon;
    double momentum;
    int    spatial;
} BatchNormalizationOpParam;


//X scale B mean var
void OpBatchNormalization1(void *op_param, float X[1][16][416][416], float scale[16], float B[16], float mean[16], float var[16], float Y[1][16][416][416], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 16;
            int Y_h = 416;
            int Y_w = 416;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
//                    ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	// norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }

//            for (n=0; n<Y_n; n++) {
//                for (c=0; c<Y_c; c++) {
//                    sum = 0.0;
//                    for (h=0; h<Y_h; h++) {
//                        for (w=0; w<Y_w; w++) {
//                            sum += X[n][c][h][w];
//                        }
//                    }
//                    ave /= (h * w);
//                    ave = momentum * ave + (1-momentum) * mean[c];
//                    sigma2 = 0.0;
//                    for (h=0; h<Y_h; h++) {
//                        for (w=0; w<Y_w; w++) {
//                            sigma2 += pow((X[n][c][h][w] - ave), 2);
//                        }
//                    }
//                    sigma2 /= (h * w);
//                    for (h=0; h<Y_h; h++) {
//                        for (w=0; w<Y_w; w++) {
//                            norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
//                            Y[n][c][h][w] = scale[c] * norm + B[c];
//                        }
//                    }
//                }
//            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}







void OpBatchNormalization2(void *op_param, float X[1][32][208][208], float scale[32], float B[32], float mean[32], float var[32], float Y[1][32][208][208], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 32;
            int Y_h = 208;
            int Y_w = 208;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	// norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}






void OpBatchNormalization3(void *op_param, float X[1][64][104][104], float scale[64], float B[64], float mean[64], float var[64], float Y[1][64][104][104], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 64;
            int Y_h = 104;
            int Y_w = 104;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
//                    ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	// norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}







void OpBatchNormalization4(void *op_param, float X[1][128][52][52], float scale[128], float B[128], float mean[128], float var[128], float Y[1][128][52][52], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 128;
            int Y_h = 52;
            int Y_w = 52;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                            // norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}






void OpBatchNormalization5(void *op_param, float X[1][256][26][26], float scale[256], float B[256], float mean[256], float var[256], float Y[1][256][26][26], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 256;
            int Y_h = 26;
            int Y_w = 26;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	//norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}






void OpBatchNormalization6(void *op_param, float X[1][512][13][13], float scale[512], float B[512], float mean[512], float var[512], float Y[1][512][13][13], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 512;
            int Y_h = 13;
            int Y_w = 13;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	// norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}







void OpBatchNormalization7(void *op_param, float X[1][1024][13][13], float scale[1024], float B[1024], float mean[1024], float var[1024], float Y[1][1024][13][13], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 1024;
            int Y_h = 13;
            int Y_w = 13;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                        	// norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}







void OpBatchNormalization8(void *op_param, float X[1][512][13][13], float scale[512], float B[512], float mean[512], float var[512], float Y[1][512][13][13], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 512;
            int Y_h = 13;
            int Y_w = 13;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                            // norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}







void OpBatchNormalization9(void *op_param, float X[1][125][13][13], float scale[125], float B[125], float mean[125], float var[125], float Y[1][125][13][13], void *inputs_params, void* outputs_params) {

            int Y_n = 1;
            int Y_c = 125;
            int Y_h = 13;
            int Y_w = 13;

            const double epsilon =  0.0010000000474974513;
            const double momentum = 0.9;
            const int    spatial =  1;

            int n;
            int c, h, w;
            double sum;
            double ave;
            double sigma2;
            double norm;

#if 1 // spatial is true
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave = sum/(h * w);
                    //ave /= (h * w);
                    // ave = momentum * ave + (1-momentum) * mean[c];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                        	norm = (X[n][c][h][w] - ave) / sqrt(sigma2 + epsilon);
                            // norm = (X[n][c][h][w] - mean[c]) / sqrt(var[c]+epsilon);
                            Y[n][c][h][w] = scale[c] * norm + B[c];
                        }
                    }
                }
            }
#else // spatial is false
            for (n=0; n<Y_n; n++) {
                for (c=0; c<Y_c; c++) {
                    sum = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sum += X[n][c][h][w];
                        }
                    }
                    ave /= (h * w);
                    ave = momentum * ave + (1-momentum) * mean[c][h][w];
                    sigma2 = 0.0;
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            sigma2 += pow((X[n][c][h][w] - ave), 2);
                        }
                    }
                    sigma2 /= (h * w);
                    for (h=0; h<Y_h; h++) {
                        for (w=0; w<Y_w; w++) {
                            norm = (X[n][c][h][w] - mean[c][h][w]) / sqrt(var[c][h][w]+epsilon);
                            Y[n][c][h][w] = scale[c][h][w] * norm + B[c][h][w];
                        }
                    }
                }
            }
#endif // spatial

}

