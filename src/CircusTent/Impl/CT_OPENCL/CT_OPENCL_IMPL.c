/*
 * TODO: __CT_OPENCL_IMPL_C_IMPL_C
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

// FIXME:
// -------------------------
#define __CL_ENABLE_EXCEPTIONS // FIXME:
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
// -------------------------

#include <stdint.h>

/* OpenCL Benchmark Implementations
 *
 * Benchmark implementations are in the form:
 *
 * void BENCHTYPE_ATOMTYPE( uint64_t *ARRAY, uint64_t *IDX,
 *                          unsigned long long iters,
 *                          unsigned long long pes )
 *
 */

// FIXME:
void RAND_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    uint64_t i      = 0;
    uint64_t start  = 0;

    // FIXME: Do I need to be making kernels?
    #pragma ocl parallel private(start, i) {
        start = (uint64_t)(get_global_id(0)) * iters; // FIXME: get_global_id might be wrong func for getting thread num
        for (i=start; i<(start+iters); i++) {
            __atomic_fetch_add( &ARRAY[IDX[i]], (uint64_t)(0x1), __ATOMIC_RELAXED ); // FIXME:
        }
    }
}

void RAND_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:

}

void STRIDE1_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void STRIDE1_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void STRIDEN_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void STRIDEN_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void PTRCHASE_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void PTRCHASE_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void SG_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void SG_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void CENTRAL_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void CENTRAL_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void SCATTER_ADD(

) {
    // todo:
}

void SCATTER_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void GATHER_ADD(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

void GATHER_CAS(
    uint64_t *restrict ARRAY,
    uint64_t *restrict IDX,
    uint64_t iters,
    uint64_t pes
) {
    // todo:
}

// ==============================================================
// EOF
