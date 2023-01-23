/*
 * _CT_CUDA_IMPL_CU_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_CUDA.cuh"

__global__ void RAND_ADD(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));
    
    for(i = start; i < (start + iters_per_thread); i++) {
        ret = atomicAdd((unsigned long long int *) &ARRAY[IDX[i]], (unsigned long long int) 0x1);
    }
}

__global__ void RAND_CAS(uint64_t* __restrict__ ARRAY, uint64_t* IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    for(i = start; i < (start + iters_per_thread); i++) {
        ret = atomicCAS((unsigned long long int *) &ARRAY[IDX[i]], (unsigned long long int) ARRAY[IDX[i]], (unsigned long long int) ARRAY[IDX[i]]);
    }
}

__global__ void STRIDE1_ADD(uint64_t* __restrict__ ARRAY, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    for(i = start; i < (start + iters_per_thread); i++) {
        ret = atomicAdd((unsigned long long int *) &ARRAY[i], (unsigned long long int) 0x1);
    }
}

__global__ void STRIDE1_CAS(uint64_t* __restrict__ ARRAY, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    for(i = start; i < (start + iters_per_thread); i++) {
        ret = atomicCAS((unsigned long long int *) &ARRAY[i], (unsigned long long int) ARRAY[i], (unsigned long long int) ARRAY[i]);
    }
}

__global__ void STRIDEN_ADD(uint64_t* __restrict__ ARRAY, uint64_t iters, uint64_t stride) {
    
    uint64_t i, ret;
    uint64_t num_threads =  (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) ((blockIdx.x * iters) + (threadIdx.x * (iters / num_threads))) * stride;

    for(i = start; i < (start + (iters_per_thread * stride)); i += stride) {
        ret = atomicAdd((unsigned long long int *) &ARRAY[i], (unsigned long long int) 0x1);
    }
}

__global__ void STRIDEN_CAS(uint64_t* __restrict__ ARRAY, uint64_t iters, uint64_t stride) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) ((blockIdx.x * iters) + (threadIdx.x * (iters / num_threads))) * stride;

    for(i = start; i < (start + (iters_per_thread * stride)); i += stride) {
        ret = atomicCAS((unsigned long long int *) &ARRAY[i], (unsigned long long int) ARRAY[i], (unsigned long long int) ARRAY[i]);
    }
}

__global__ void CENTRAL_ADD(uint64_t* __restrict__ ARRAY, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));

    for(i = 0; i < iters_per_thread; i++) {
        ret = atomicAdd((unsigned long long int *) &ARRAY[0], (unsigned long long int) 0x1);
    }
}

__global__ void CENTRAL_CAS(uint64_t* __restrict__ ARRAY, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));

    for(i = 0; i < iters_per_thread; i++) {
        ret = atomicCAS((unsigned long long int *) &ARRAY[0], (unsigned long long int) ARRAY[0], (unsigned long long int) ARRAY[0]);
    }
}

/* Note that the PTRCHASE kernels utilize only a single thread per thread block. As the *
 * iterations for a given thread block are not independent, utilizing multiple threads  *
 * per block would destroy the semantics of a pointer chasing operation.                */
__global__ void PTRCHASE_ADD(uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i;
    uint64_t start = (uint64_t) (blockIdx.x * iters);

    for(i = 0; i < iters; i++) {
        start = atomicAdd((unsigned long long int *) &IDX[start], (unsigned long long int) 0x0);
    }
}

/* Note that the PTRCHASE kernels utilize only a single thread per thread block. As the *
 * iterations for a given thread block are not independent, utilizing multiple threads  *
 * per block would destroy the semantics of a pointer chasing operation.                */
__global__ void PTRCHASE_CAS(uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i;
    uint64_t start = (uint64_t) (blockIdx.x * iters);

    for(i = 0; i < iters; i++) {
        start = atomicCAS((unsigned long long int *) &IDX[start], (unsigned long long int) IDX[start], (unsigned long long int) IDX[start]);
    }
}

__global__ void SG_ADD(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t src, dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    src =  0x0;
    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        src = atomicAdd((unsigned long long int *) &IDX[i], (unsigned long long int) 0x0);
        dest = atomicAdd((unsigned long long int *) &IDX[i+1], (unsigned long long int) 0x0);
        val = atomicAdd((unsigned long long int *) &ARRAY[src], (unsigned long long int) 0x1);
        ret = atomicAdd((unsigned long long int *) &ARRAY[dest], (unsigned long long int) val);
    }
}

__global__ void SG_CAS(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t src, dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    src =  0x0;
    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        src = atomicCAS((unsigned long long int *) &IDX[i], (unsigned long long int) IDX[i], (unsigned long long int) IDX[i]);
        dest = atomicCAS((unsigned long long int *) &IDX[i+1], (unsigned long long int) IDX[i+1], (unsigned long long int) IDX[i+1]);
        val = atomicCAS((unsigned long long int *) &ARRAY[src], (unsigned long long int) ARRAY[src], (unsigned long long int) ARRAY[src]);
        ret = atomicCAS((unsigned long long int *) &ARRAY[dest], (unsigned long long int) ARRAY[dest], (unsigned long long int) val);
    }
}

__global__ void SCATTER_ADD(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        dest = atomicAdd((unsigned long long int *) &IDX[i+1], (unsigned long long int) 0x0);
        val = atomicAdd((unsigned long long int *) &ARRAY[i], (unsigned long long int) 0x1);
        ret = atomicAdd((unsigned long long int *) &ARRAY[dest], (unsigned long long int) val);
    }
}

__global__ void SCATTER_CAS(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        dest = atomicCAS((unsigned long long int *) &IDX[i+1], (unsigned long long int) IDX[i+1], (unsigned long long int) IDX[i+1]);
        val = atomicCAS((unsigned long long int *) &ARRAY[i], (unsigned long long int) ARRAY[i], (unsigned long long int) ARRAY[i]);
        ret = atomicCAS((unsigned long long int *) &ARRAY[dest], (unsigned long long int) ARRAY[dest], (unsigned long long int) val);
    }
}

__global__ void GATHER_ADD(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        dest = atomicAdd((unsigned long long int *) &IDX[i+1], (unsigned long long int) 0x0);
        val = atomicAdd((unsigned long long int *) &ARRAY[dest], (unsigned long long int) 0x1);
        ret = atomicAdd((unsigned long long int *) &ARRAY[i], (unsigned long long int) val);
    }
}

__global__ void GATHER_CAS(uint64_t* __restrict__ ARRAY, uint64_t* __restrict__ IDX, uint64_t iters) {
    
    uint64_t i, ret;
    uint64_t dest, val;
    uint64_t num_threads = (uint64_t) blockDim.x;
    uint64_t iters_per_thread = (uint64_t) ((threadIdx.x == num_threads - 1) ?
                                            (iters / num_threads) + (iters % num_threads) : 
                                            (iters / num_threads));
    uint64_t start = (uint64_t) (blockIdx.x * iters) + (threadIdx.x * (iters / num_threads));

    dest = 0x0;
    val =  0x0;

    for(i = start; i < (start + iters_per_thread); i++) {
        dest = atomicCAS((unsigned long long int *) &IDX[i+1], (unsigned long long int) IDX[i+1], (unsigned long long int) IDX[i+1]);
        val = atomicCAS((unsigned long long int *) &ARRAY[dest], (unsigned long long int) ARRAY[dest], (unsigned long long int) ARRAY[dest]);
        ret = atomicCAS((unsigned long long int *) &ARRAY[i], (unsigned long long int) ARRAY[i], (unsigned long long int) val);
    }
}