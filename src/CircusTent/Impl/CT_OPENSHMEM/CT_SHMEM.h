//
// _CT_SHMEM_H_
//
// Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_SHMEM
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent OpenSHMEM Implementation
 *
 */

#ifdef _ENABLE_OPENSHMEM_

#ifndef _CT_SHMEM_H_
#define _CT_SHMEM_H_

#include <cstdlib>
#include <mpp/shmem.h>
#include <ctime>

#include "CircusTent/CTBaseImpl.h"

// Benchmark Prototypes
extern "C" {
/// RAND AMO ADD Benchmark
void RAND_ADD( uint64_t *ARRAY,
               uint64_t *IDX,
               int *TARGET,
               uint64_t iters,
               uint64_t pes );

/// RAND AMO CAS Benchmark
void RAND_CAS( uint64_t *ARRAY,
               uint64_t *IDX,
               int *TARGET,
               uint64_t iters,
               uint64_t pes );

/// STRIDE1 AMO ADD Benchmark
void STRIDE1_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// STRIDE1 AMO CAS Benchmark
void STRIDE1_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// STRIDEN AMO ADD Benchmark
void STRIDEN_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride );

/// STRIDEN AMO CAS Benchmark
void STRIDEN_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride );

/// PTRCHASE AMO ADD Benchmark
void PTRCHASE_ADD( uint64_t *ARRAY,
                   uint64_t *IDX,
                   int *TARGET,
                   uint64_t iters,
                   uint64_t pes );

/// PTRCHASE AMO CAS Benchmark
void PTRCHASE_CAS( uint64_t *ARRAY,
                   uint64_t *IDX,
                   int *TARGET,
                   uint64_t iters,
                   uint64_t pes );

/// SG AMO ADD Benchmark
void SG_ADD( uint64_t *ARRAY,
             uint64_t *IDX,
             int *TARGET,
             uint64_t iters,
             uint64_t pes );

/// SG AMO CAS Benchmark
void SG_CAS( uint64_t *ARRAY,
             uint64_t *IDX,
             int *TARGET,
             uint64_t iters,
             uint64_t pes );

/// CENTRAL AMO ADD Benchmark
void CENTRAL_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// CENTRAL AMO CAS Benchmark
void CENTRAL_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// SCATTER AMO ADD Benchmark
void SCATTER_ADD( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// SCATTER AMO CAS Benchmark
void SCATTER_CAS( uint64_t *ARRAY,
                  uint64_t *IDX,
                  int *TARGET,
                  uint64_t iters,
                  uint64_t pes );

/// GATHER AMO ADD Benchmark
void GATHER_ADD( uint64_t *ARRAY,
                 uint64_t *IDX,
                 int *TARGET,
                 uint64_t iters,
                 uint64_t pes );

/// GATHER AMO CAS Benchmark
void GATHER_CAS( uint64_t *ARRAY,
                 uint64_t *IDX,
                 int *TARGET,
                 uint64_t iters,
                 uint64_t pes );

}


class CT_SHMEM : public CTBaseImpl{
private:
  uint64_t *Array;          ///< CT_OPENSHMEM: Data array
  uint64_t *Idx;            ///< CT_OPENSHMEM: Index array
  int *Target;              ///< CT_OPENSHMEM: Target PE array
  uint64_t memSize;         ///< CT_OPENSHMEM: Memory size (in bytes)
  uint64_t pes;             ///< CT_OPENSHMEM: Number of processing elements
  uint64_t iters;           ///< CT_OPENSHMEM: Number of iterations per thread
  uint64_t elems;           ///< CT_OPENSHMEM: Number of u8 elements
  uint64_t stride;          ///< CT_OPENSHMEM: Stride in elements

public:
  /// CircusTent OpenSHMEM constructor
  CT_SHMEM(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent OpenSHMEM destructor
  ~CT_SHMEM();

  /// CircusTent OpenSHMEM exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenSHMEM data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent OpenSHMEM data free function
  virtual bool FreeData() override;
};

#endif  // _CT_SHMEM_H_
#endif  // _ENABLE_OPENSHMEM_

// EOF
