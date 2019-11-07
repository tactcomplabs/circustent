/*
 * _CT_SHMEM_IMPL_C_
 *
 * Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <shmem.h>
#include <stdint.h>


/* OpenSHMEM Benchmark Implementations
 *
 * Benchmark implementations are in the form:
 *
 * void BENCHTYPE_ATOMTYPE( uint64_t *ARRAY, uint64_t *IDX, int *TARGET,
 *                          unsigned long long iters,
 *                          unsigned long long pes )
 *
 * ARRAY and IDX are in the SHMEM symmetric heap
 * TARGET is a local PE array
 *
 */

void RAND_ADD( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               int *restrict TARGET,
               uint64_t iters,
               uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    shmem_get8(&start,&IDX[i],1,TARGET[i]);
    start = shmem_ulong_atomic_fetch_add(&ARRAY[start],(uint64_t)(0x1),TARGET[i]);
  }
}

void RAND_CAS( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               int *restrict TARGET,
               uint64_t iters,
               uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    shmem_get8(&start,&IDX[i],1,TARGET[i]);
    start = shmem_ulong_atomic_compare_swap(&ARRAY[start],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                             TARGET[i]);
  }
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_fetch_add(&ARRAY[i],(uint64_t)(0xF),TARGET[i]);
  }
}

void STRIDE1_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_compare_swap(&ARRAY[i],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                             TARGET[i]);
  }
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t idx    = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_fetch_add(&ARRAY[idx],(uint64_t)(0xF),TARGET[i]);
    idx += stride;
  }
}

void STRIDEN_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride ){
  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t idx    = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_compare_swap(&ARRAY[idx],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                             TARGET[i]);
    idx += stride;
  }
}

void PTRCHASE_ADD( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   int *restrict TARGET,
                   uint64_t iters,
                   uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_fetch_add(&IDX[start],(uint64_t)(0x00ull),TARGET[i]);
  }
}

void PTRCHASE_CAS( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   int *restrict TARGET,
                   uint64_t iters,
                   uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_compare_swap(&IDX[start],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                             TARGET[i]);
  }
}

void SG_ADD( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             int *restrict TARGET,
             uint64_t iters,
             uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t src    = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    src   = shmem_ulong_atomic_fetch_add(&IDX[i],(uint64_t)(0x00ull),TARGET[i]);
    dest  = shmem_ulong_atomic_fetch_add(&IDX[+1],(uint64_t)(0x00ull),TARGET[i]);
    val   = shmem_ulong_atomic_fetch_add(&ARRAY[src],(uint64_t)(0x01ull),TARGET[i]);
    start = shmem_ulong_atomic_fetch_add(&ARRAY[dest], val, TARGET[i] );
  }
}

void SG_CAS( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             int *restrict TARGET,
             uint64_t iters,
             uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t src    = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    src   = shmem_ulong_atomic_compare_swap(&IDX[i],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    dest  = shmem_ulong_atomic_compare_swap(&IDX[i+1],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    val   = shmem_ulong_atomic_compare_swap(&ARRAY[src],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    start = shmem_ulong_atomic_compare_swap(&ARRAY[dest],
                                            (uint64_t)(0x00),
                                            val,
                                            TARGET[i]);
  }
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){
  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_fetch_add(&ARRAY[0],(uint64_t)(0x1),TARGET[i]);
  }
}

void CENTRAL_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){
  uint64_t i      = 0;
  uint64_t start  = 0;

  for( i=0; i<iters; i++ ){
    start = shmem_ulong_atomic_compare_swap(&ARRAY[0],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                             TARGET[i]);
  }
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    dest  = shmem_ulong_atomic_fetch_add(&IDX[+1],(uint64_t)(0x00ull),TARGET[i]);
    val   = shmem_ulong_atomic_fetch_add(&ARRAY[i],(uint64_t)(0x01ull),TARGET[i]);
    start = shmem_ulong_atomic_fetch_add(&ARRAY[dest], val, TARGET[i] );
  }
}

void SCATTER_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    dest  = shmem_ulong_atomic_compare_swap(&IDX[i+1],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    val   = shmem_ulong_atomic_compare_swap(&ARRAY[i],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    start = shmem_ulong_atomic_compare_swap(&ARRAY[dest],
                                            (uint64_t)(0x00),
                                            val,
                                            TARGET[i]);
  }
}

void GATHER_ADD( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 int *restrict TARGET,
                 uint64_t iters,
                 uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    dest  = shmem_ulong_atomic_fetch_add(&IDX[+1],(uint64_t)(0x00ull),TARGET[i]);
    val   = shmem_ulong_atomic_fetch_add(&ARRAY[dest],(uint64_t)(0x01ull),TARGET[i]);
    start = shmem_ulong_atomic_fetch_add(&ARRAY[i], val, TARGET[i] );
  }
}

void GATHER_CAS( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 int *restrict TARGET,
                 uint64_t iters,
                 uint64_t pes ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;

  for( i=0; i<iters; i++ ){
    dest  = shmem_ulong_atomic_compare_swap(&IDX[i+1],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    val   = shmem_ulong_atomic_compare_swap(&ARRAY[dest],
                                            (uint64_t)(0x00),
                                            (uint64_t)(0x00),
                                            TARGET[i]);
    start = shmem_ulong_atomic_compare_swap(&ARRAY[i],
                                            (uint64_t)(0x00),
                                            val,
                                            TARGET[i]);
  }
}



/* EOF */
