//
// _CT_PTHREADS_H_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_PTHREADS
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent Pthreads Implementation
 *
 */

#ifdef _ENABLE_PTHREADS_

#ifndef _CT_PTHREADS_H_
#define _CT_PTHREADS_H_

#include <cstdlib>
#include <pthread.h>
#include <ctime>

#include "CircusTent/CTBaseImpl.h"

// Benchmark Prototypes
extern "C" {

/// RAND AMO ADD Benchmark
void RAND_ADD( uint64_t *ARRAY,
               uint64_t *IDX,
               uint64_t iters,
               uint64_t pes,
               pthread_barrier_t *barrier,
               double *start_time );

/// RAND AMO CAS Benchmark
void RAND_CAS( uint64_t *ARRAY,
               uint64_t *IDX,
               uint64_t iters,
               uint64_t pes,
               pthread_barrier_t *barrier,
               double *start_time );

/// STRIDE1 AMO ADD Benchmark
void STRIDE1_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// STRIDE1 AMO CAS Benchmark
void STRIDE1_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// STRIDEN AMO ADD Benchmark
void STRIDEN_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// STRIDEN AMO CAS Benchmark
void STRIDEN_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// PTRCHASE AMO ADD Benchmark
void PTRCHASE_ADD( uint64_t *ARRAY,
                   uint64_t *IDX,
                   uint64_t iters,
                   uint64_t pes,
                   pthread_barrier_t *barrier,
                   double *start_time );

/// PTRCHASE AMO CAS Benchmark
void PTRCHASE_CAS( uint64_t *ARRAY,
                   uint64_t *IDX,
                   uint64_t iters,
                   uint64_t pes,
                   pthread_barrier_t *barrier,
                   double *start_time );

/// SG AMO ADD Benchmark
void SG_ADD( uint64_t *ARRAY,
             uint64_t *IDX,
             uint64_t iters,
             uint64_t pes,
             pthread_barrier_t *barrier,
             double *start_time );

/// SG AMO CAS Benchmark
void SG_CAS( uint64_t *ARRAY,
             uint64_t *IDX,
             uint64_t iters,
             uint64_t pes,
             pthread_barrier_t *barrier,
             double *start_time );

/// CENTRAL AMO ADD Benchmark
void CENTRAL_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// CENTRAL AMO CAS Benchmark
void CENTRAL_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// SCATTER AMO ADD Benchmark
void SCATTER_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// SCATTER AMO CAS Benchmark
void SCATTER_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  pthread_barrier_t *barrier,
                  double *start_time );

/// GATHER AMO ADD Benchmark
void GATHER_ADD( uint64_t *ARRAY,
                 uint64_t *IDX,
                 uint64_t iters,
                 uint64_t pes,
                 pthread_barrier_t *barrier,
                 double *start_time );

/// GATHER AMO CAS Benchmark
void GATHER_CAS( uint64_t *ARRAY,
                 uint64_t *IDX,
                 uint64_t iters,
                 uint64_t pes,
                 pthread_barrier_t *barrier,
                 double *start_time );

}

class CT_PTHREADS : public CTBaseImpl{
private:
  uint64_t *Array;          ///< CT_PTHREADS: Data array
  uint64_t *Idx;            ///< CT_PTHREADS: Index array
  uint64_t memSize;         ///< CT_PTHREADS: Memory size (in bytes)
  uint64_t pes;             ///< CT_PTHREADS: Number of processing elements
  uint64_t iters;           ///< CT_PTHREADS: Number of iterations per thread
  uint64_t elems;           ///< CT_PTHREADS: Number of u8 elements
  uint64_t stride;          ///< CT_PTHREADS: Stride in elements

public:
  /// CircusTent Pthreads constructor
  CT_PTHREADS(CTBaseImpl::CTBenchType B,
              CTBaseImpl::CTAtomType A);

  /// CircusTent Pthreads destructor
  ~CT_PTHREADS();

  /// CircusTent Pthreads exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent Pthreads data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent Pthreads data free function
  virtual bool FreeData() override;
};

#endif  // _CT_PTHREADS_H_
#endif  // _ENABLE_PTHREADS_

// EOF
