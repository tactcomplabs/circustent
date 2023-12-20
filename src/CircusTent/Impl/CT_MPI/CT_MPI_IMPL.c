/*
 * _CT_MPI_IMPL_C_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

/* MPI Benchmark Implementations
 *
 * Benchmark implementations are in the form:
 *
 * void BENCHTYPE_ATOMTYPE( uint64_t *ARRAY, uint64_t *IDX, int *TARGET,
 *                          unsigned long long iters,
 *                          unsigned long long pes,
 *                          MPI_Win ARRAY_WIN, MPI_Win IDX_WIN )
 *
 * ARRAY and IDX are in the MPI windows
 * TARGET is a local PE array
 *
 */

void RAND_ADD( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               int *restrict TARGET,
               uint64_t iters,
               uint64_t pes,
               MPI_Win AWin,
               MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Get((unsigned long *)(&start),1,MPI_UNSIGNED_LONG,TARGET[i],
            ((&IDX[i])-(&IDX[0])),1,MPI_UNSIGNED_LONG,IWin);
    MPI_Win_flush_local(TARGET[i], IWin);

    MPI_Fetch_and_op((unsigned long *)(&start),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[start])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void RAND_CAS( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               int *restrict TARGET,
               uint64_t iters,
               uint64_t pes,
               MPI_Win AWin,
               MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Get((unsigned long *)(&start),1,MPI_UNSIGNED_LONG,TARGET[i],
            ((&IDX[i])-(&IDX[0])),1,MPI_UNSIGNED_LONG,IWin);
    MPI_Win_flush_local(TARGET[i], IWin);

    MPI_Compare_and_swap((unsigned long *)(&start),(unsigned long *)(&start),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[start])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    // buffer use not standard but appears benign.
    //
    // This technically causes undefined behavior, origin buffer and result buffer
    // aren't supposed to be the same per documentation.
    // Hypothesis for what is happening:
    //  - array value is copied into start,
    //  - start and array value are added, (array value is doubled in place)
    //  - start is returned to the rank that called the operation.
    // Doesn't break program and still implements the desired AMO.
    
    MPI_Fetch_and_op((unsigned long *)(&start),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[i])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void STRIDE1_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&start),(unsigned long *)(&start),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[i])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride,
                  MPI_Win AWin,
                  MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t idx    = 0;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    MPI_Fetch_and_op((unsigned long *)(&start),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[idx])-(&ARRAY[0])),MPI_SUM,AWin);
    idx += stride;
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void STRIDEN_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride,
                  MPI_Win AWin,
                  MPI_Win IWin ){
  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t idx    = 0;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&start),(unsigned long *)(&start),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[idx])-(&ARRAY[0])),AWin);
    idx += stride;
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void PTRCHASE_ADD( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   int *restrict TARGET,
                   uint64_t iters,
                   uint64_t pes,
                   MPI_Win AWin,
                   MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t zero   = 0;

  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Fetch_and_op((unsigned long *)(&zero),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&IDX[start])-(&IDX[0])),MPI_SUM,IWin);
    MPI_Win_flush(TARGET[i], IWin);
  }
  MPI_Win_unlock_all(IWin);
}

void PTRCHASE_CAS( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   int *restrict TARGET,
                   uint64_t iters,
                   uint64_t pes,
                   MPI_Win AWin,
                   MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t zero   = 0;

  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&start),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&IDX[start])-(&IDX[0])),IWin);
    MPI_Win_flush(TARGET[i], IWin);
  }
  MPI_Win_unlock_all(IWin);
}

void SG_ADD( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             int *restrict TARGET,
             uint64_t iters,
             uint64_t pes,
             MPI_Win AWin,
             MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t src    = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Fetch_and_op((unsigned long *)(&zero),(unsigned long *)(&src),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&IDX[i])-(&IDX[0])),MPI_SUM,IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Fetch_and_op((unsigned long *)(&zero),(unsigned long *)(&dest),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&IDX[i+1])-(&IDX[0])),MPI_SUM,IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Fetch_and_op((unsigned long *)(&one),(unsigned long *)(&val),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[src])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Fetch_and_op((unsigned long *)(&val),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[dest])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void SG_CAS( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             int *restrict TARGET,
             uint64_t iters,
             uint64_t pes,
             MPI_Win AWin,
             MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t src    = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&src),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&IDX[i])-(&IDX[0])),IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&dest),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&IDX[i+1])-(&IDX[0])),IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&val),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[src])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&val),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[dest])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){
  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    // Would appear that it targets the first element 
    // at the target rank, not the first element of the 
    // shared data array. This is as per CT documentation, but not like YGM.
    
    MPI_Fetch_and_op((unsigned long *)(&one),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     (0),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void CENTRAL_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){
  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&one),(unsigned long *)(&start),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         (0),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Fetch_and_op((unsigned long *)(&zero),(unsigned long *)(&dest),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&IDX[i+1])-(&IDX[0])),MPI_SUM,IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Fetch_and_op((unsigned long *)(&one),(unsigned long *)(&val),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[i])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Fetch_and_op((unsigned long *)(&val),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[dest])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void SCATTER_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  int *restrict TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  MPI_Win AWin,
                  MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&dest),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&IDX[i+1])-(&IDX[0])),IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&one),
                         (unsigned long *)(&val),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[i])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&val),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[dest])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void GATHER_ADD( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 int *restrict TARGET,
                 uint64_t iters,
                 uint64_t pes,
                 MPI_Win AWin,
                 MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;
  uint64_t one    = 0x01ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Fetch_and_op((unsigned long *)(&zero),(unsigned long *)(&dest),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&IDX[i+1])-(&IDX[0])),MPI_SUM,IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Fetch_and_op((unsigned long *)(&one),(unsigned long *)(&val),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[dest])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Fetch_and_op((unsigned long *)(&val),(unsigned long *)(&start),
                     MPI_UNSIGNED_LONG,TARGET[i],
                     ((&ARRAY[i])-(&ARRAY[0])),MPI_SUM,AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

void GATHER_CAS( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 int *restrict TARGET,
                 uint64_t iters,
                 uint64_t pes,
                 MPI_Win AWin,
                 MPI_Win IWin ){

  uint64_t i      = 0;
  uint64_t start  = 0;
  uint64_t dest   = 0;
  uint64_t val    = 0;
  uint64_t zero   = 0x00ull;

  MPI_Win_lock_all(0, AWin);
  MPI_Win_lock_all(0, IWin);

  for( i=0; i<iters; i++ ){
    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&dest),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&IDX[i+1])-(&IDX[0])),IWin);
    MPI_Win_flush(TARGET[i], IWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&zero),
                         (unsigned long *)(&val),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[dest])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);

    MPI_Compare_and_swap((unsigned long *)(&zero),(unsigned long *)(&val),
                         (unsigned long *)(&start),
                         MPI_UNSIGNED_LONG,TARGET[i],
                         ((&ARRAY[i])-(&ARRAY[0])),AWin);
    MPI_Win_flush(TARGET[i], AWin);
  }
  MPI_Win_unlock_all(AWin);
  MPI_Win_unlock_all(IWin);
}

/* EOF */
