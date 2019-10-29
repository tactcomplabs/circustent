//
// _CT_OMP_H_
//
// Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_OMP
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent OpenMP Implementation
 *
 */

#ifdef _ENABLE_OMP_

#ifndef _CT_OMP_H_
#define _CT_OMP_H_

#include <cstdlib>
#include <omp.h>
#include <ctime>

#include "CircusTent/CTBaseImpl.h"

// Benchmark Prototypes
extern "C" {
/// RAND AMO ADD Benchmark
void RAND_ADD( uint64_t *ARRAY,
               uint64_t *IDX,
               uint64_t iters,
               uint64_t pes );

/// RAND AMO CAS Benchmark
void RAND_CAS( uint64_t *ARRAY,
               uint64_t *IDX,
               uint64_t iters,
               uint64_t pes );

/// STRIDE1 AMO ADD Benchmark
void STRIDE1_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// STRIDE1 AMO CAS Benchmark
void STRIDE1_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// STRIDEN AMO ADD Benchmark
void STRIDEN_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride );

/// STRIDEN AMO CAS Benchmark
void STRIDEN_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride );

/// PTRCHASE AMO ADD Benchmark
void PTRCHASE_ADD( uint64_t *ARRAY,
                   uint64_t *IDX,
                   uint64_t iters,
                   uint64_t pes );

/// PTRCHASE AMO CAS Benchmark
void PTRCHASE_CAS( uint64_t *ARRAY,
                   uint64_t *IDX,
                   uint64_t iters,
                   uint64_t pes );

/// SG AMO ADD Benchmark
void SG_ADD( uint64_t *ARRAY,
             uint64_t *IDX,
             uint64_t iters,
             uint64_t pes );

/// SG AMO CAS Benchmark
void SG_CAS( uint64_t *ARRAY,
             uint64_t *IDX,
             uint64_t iters,
             uint64_t pes );

/// CENTRAL AMO ADD Benchmark
void CENTRAL_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// CENTRAL AMO CAS Benchmark
void CENTRAL_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// SCATTER AMO ADD Benchmark
void SCATTER_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// SCATTER AMO CAS Benchmark
void SCATTER_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// GATHER AMO ADD Benchmark
void GATHER_ADD( uint64_t *ARRAY,
                 uint64_t *IDX,
                 uint64_t iters,
                 uint64_t pes );

/// GATHER AMO CAS Benchmark
void GATHER_CAS( uint64_t *ARRAY,
                 uint64_t *IDX,
                 uint64_t iters,
                 uint64_t pes );

}


class CT_OMP : public CTBaseImpl{
private:
  uint64_t *Array;          ///< CT_OMP: Data array
  uint64_t *Idx;            ///< CT_OMP: Index array
  uint64_t memSize;         ///< CT_OMP: Memory size (in bytes)
  uint64_t pes;             ///< CT_OMP: Number of processing elements
  uint64_t iters;           ///< CT_OMP: Number of iterations per thread
  uint64_t elems;           ///< CT_OMP: Number of u8 elements
  uint64_t stride;          ///< CT_OMP: Stride in elements

public:
  /// CircusTent OpenMP constructor
  CT_OMP(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent OpenMP destructor
  ~CT_OMP();

  /// CircusTent OpenMP exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenMP data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent OpenMP data free function
  virtual bool FreeData() override;
};

#endif  // _CT_OMP_H_
#endif  // _ENABLE_OMP_

// EOF
