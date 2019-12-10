#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include "out_c/include/numpy.h"

uint8_t vi_input[1][128][128][3];
uint8_t vi_MobilenetV1_Predictions_Reshape_1[1][1001];


int qumico( uint8_t [1][128][128][3], uint8_t [1][1001] );

int main( ) {
    int max=0;
    int max_idx = 0;

    FILE *fp;
    char *buf = NULL;
    size_t buflen;
    int  idx_cnt;

    NUMPY_HEADER nph_input;
    if (load_from_numpy( vi_input, "input.npy", 49152, &nph_input ) != 0) {
        return 1;
    }

    qumico( vi_input, vi_MobilenetV1_Predictions_Reshape_1);
    for (int i=0; i<1001; i++) {
        printf("%d:", vi_MobilenetV1_Predictions_Reshape_1[0][i] );
        if (max < vi_MobilenetV1_Predictions_Reshape_1[0][i]) {
            max = vi_MobilenetV1_Predictions_Reshape_1[0][i];
	    max_idx = i;
        }
    }
    printf("\n\nmax: %d\n", max_idx );

    idx_cnt = 0;
    fp = fopen( "labels.txt", "r" );
    while(feof(fp) == 0) {
        getline(&buf, &buflen, fp);
        if (idx_cnt == max_idx) {
            printf("%s", buf);
            break;
        }
        idx_cnt++;
    }

    return 0;
}

