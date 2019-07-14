#pragma once
#include "qumico_type.h"

typedef struct numpy_header_tag {
    unsigned char major_version; //major version
    unsigned char minor_version; //minor version
    unsigned short header_len;
    QMC_DTYPE descr;
    bool fortran_order;
    int shape[4];
} NUMPY_HEADER;


extern const NUMPY_HEADER default_numpy_header;

int load_from_numpy(void *dp, const char *numpy_fname, int size, NUMPY_HEADER *hp);
int save_to_numpy(void *dp, const char *numpy_fname, NUMPY_HEADER *hp);

