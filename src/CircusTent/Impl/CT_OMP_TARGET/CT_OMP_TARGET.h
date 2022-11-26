//
// _CT_OMP_TARGET_H
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_OMP_TARGET
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent OpenMP Target Implementation
 *
 */

#ifdef _ENABLE_OMP_TARGET_

#ifndef _CT_OMP_TARGET_H_
#define _CT_OMP_TARGET_H_

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

/// STRIDE1 AMO ADD Benchmark
void STRIDE1_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// STRIDEN AMO ADD Benchmark
void STRIDEN_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride );

/// PTRCHASE AMO ADD Benchmark
void PTRCHASE_ADD( uint64_t *ARRAY,
                   uint64_t *IDX,
                   uint64_t iters,
                   uint64_t pes );

/// SG AMO ADD Benchmark
void SG_ADD( uint64_t *ARRAY,
             uint64_t *IDX,
             uint64_t iters,
             uint64_t pes );

/// CENTRAL AMO ADD Benchmark
void CENTRAL_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// SCATTER AMO ADD Benchmark
void SCATTER_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  uint64_t iters,
                  uint64_t pes );

/// GATHER AMO ADD Benchmark
void GATHER_ADD( uint64_t *ARRAY,
                 uint64_t *IDX,
                 uint64_t iters,
                 uint64_t pes );
}

class CT_OMP_TARGET : public CTBaseImpl{
private:
  uint64_t *Array;          ///< CT_OMP_TARGET: Data array
  uint64_t *Idx;            ///< CT_OMP_TARGET: Index array
  uint64_t memSize;         ///< CT_OMP_TARGET: Memory size (in bytes)
  uint64_t pes;             ///< CT_OMP_TARGET: Number of teams
  uint64_t iters;           ///< CT_OMP_TARGET: Number of iterations per team
  uint64_t elems;           ///< CT_OMP_TARGET: Number of u8 elements
  uint64_t stride;          ///< CT_OMP_TARGET: Stride in elements
  int deviceID;             ///< CT_OMP_TARGET: Target device id

public:
  /// CircusTent OpenMP Target constructor
  CT_OMP_TARGET(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent OpenMP Target destructor
  ~CT_OMP_TARGET();

  /// CircusTent OpenMP Target exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenMP Target data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent OpenMP Target data free function
  virtual bool FreeData() override;

  /// Function to set target device options
  bool SetDevice();
};

#endif  // _CT_OMP_TARGET_H_
#endif  // _ENABLE_OMP_TARGET_

// EOF
