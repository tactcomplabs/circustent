//
// _CT_OPENACC_H_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_OPENACC
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent OpenACC Implementation
 *
 */

#ifdef _ENABLE_OPENACC_

#ifndef _CT_OPENACC_H_
#define _CT_OPENACC_H_

#include <cstdlib>
#include <openacc.h>
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

class CT_OPENACC : public CTBaseImpl{
private:
  uint64_t *Array;             ///< CT_OPENACC: Data array
  uint64_t *Idx;               ///< CT_OPENACC: Index array
  uint64_t memSize;            ///< CT_OPENACC: Memory size (in bytes)
  uint64_t pes;                ///< CT_OPENACC: Number of gangs
  uint64_t iters;              ///< CT_OPENACC: Number of iterations per gang
  uint64_t elems;              ///< CT_OPENACC: Number of u8 elements
  uint64_t stride;             ///< CT_OPENACC: Stride in elements
  acc_device_t deviceTypeEnum; ///< CT_OPENACC: acct_device_t device type enumeration

public:
  /// CircusTent OpenACC constructor
  CT_OPENACC(CTBaseImpl::CTBenchType B,
             CTBaseImpl::CTAtomType A);

  /// CircusTent OpenACC destructor
  ~CT_OPENACC();

  /// CircusTent OpenACC exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenACC data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent OpenACC data free function
  virtual bool FreeData() override;

  /// Function to set target device options
  bool SetDevice();
};

#endif  // _CT_OPENACC_H_
#endif  // _ENABLE_OPENACC_

// EOF
