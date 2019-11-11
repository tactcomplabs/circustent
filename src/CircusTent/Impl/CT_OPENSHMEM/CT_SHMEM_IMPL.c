/*
 * _CT_SHMEM_IMPL_C_
 *
 * Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <mpp/shmem.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


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
    shmem_long_get((long *)(&start),(long *)(&IDX[i]),1,TARGET[i]);
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[start]),(long)(0x1),TARGET[i]));
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
    shmem_long_get((long *)(&start),(long *)(&IDX[i]),1,TARGET[i]);
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[start]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
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
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[i]),(long)(0xF),TARGET[i]));
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
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[i]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
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
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[idx]),(long)(0xF),TARGET[i]));
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
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[idx]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
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
    start = (uint64_t)(shmem_long_fadd((long *)(&IDX[start]),(long)(0x00ull),TARGET[i]));
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
    start = (uint64_t)(shmem_long_cswap((long *)(&IDX[start]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
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
    src   = (uint64_t)(shmem_long_fadd((long *)(&IDX[i]),(long)(0x00ull),TARGET[i]));
    dest  = (uint64_t)(shmem_long_fadd((long *)(&IDX[+1]),(long)(0x00ull),TARGET[i]));
    val   = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[src]),(long)(0x01ull),TARGET[i]));
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[dest]), (long)(val), TARGET[i]));
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
    src   = (uint64_t)(shmem_long_cswap((long *)(&IDX[i]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    dest  = (uint64_t)(shmem_long_cswap((long *)(&IDX[i+1]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    val   = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[src]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[dest]),
                                        (long)(0x00),
                                        (long )(val),
                                        TARGET[i]));
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
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[0]),(long)(0x1),TARGET[i]));
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
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[0]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
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
    dest  = (uint64_t)(shmem_long_fadd((long *)(&IDX[+1]),(long)(0x00ull),TARGET[i]));
    val   = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[i]),(long)(0x01ull),TARGET[i]));
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[dest]), (long)(val), TARGET[i]));
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
    dest  = (uint64_t)(shmem_long_cswap((long *)(&IDX[i+1]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    val   = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[i]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[dest]),
                                        (long)(0x00),
                                        (long)(val),
                                        TARGET[i]));
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
    dest  = (uint64_t)(shmem_long_fadd((long *)(&IDX[+1]),(long)(0x00ull),TARGET[i]));
    val   = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[dest]),(long)(0x01ull),TARGET[i]));
    start = (uint64_t)(shmem_long_fadd((long *)(&ARRAY[i]), (long)(val), TARGET[i]));
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
    dest  = (uint64_t)(shmem_long_cswap((long *)(&IDX[i+1]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    val   = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[dest]),
                                        (long)(0x00),
                                        (long)(0x00),
                                        TARGET[i]));
    start = (uint64_t)(shmem_long_cswap((long *)(&ARRAY[i]),
                                        (long)(0x00),
                                        (long)(val),
                                        TARGET[i]));
  }
}



/* EOF */
