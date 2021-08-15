/*
 * TODO: CT_OPENCL__H
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

/**
 * \class CT_OPENCL
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent OpenCL Implementation
 *
 */

#ifdef _ENABLE_OPENCL_      // todo

#ifndef _CT_OPENCL_H_       // todo
#define _CT_OPENCL_H_       // todo

#include <cstdlib>
#include <> // TODO: ADD OPENCL HEADER
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

class CT_OPENCL : public CTBaseImpl{
private:
  uint64_t *Array;          ///< CT_OPENCL: Data array
  uint64_t *Idx;            ///< CT_OPENCL: Index array
  uint64_t memSize;         ///< CT_OPENCL: Memory size (in bytes)
  uint64_t pes;             ///< CT_OPENCL: Number of processing elements
  uint64_t iters;           ///< CT_OPENCL: Number of iterations per team
  uint64_t elems;           ///< CT_OPENCL: Number of u8 elements
  uint64_t stride;          ///< CT_OPENCL: Stride in elements
  int deviceID;             ///< CT_OPENCL: Target device id

public:
  /// CircusTent OpenCL Target constructor
  CT_OPENCL(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent OpenCL Target destructor
  ~CT_OPENCL();

  /// CircusTent OpenCL Target exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenCL Target data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent OpenCL Target data free function
  virtual bool FreeData() override;

  /// Function to set target device options
  bool SetDevice();
};

#endif  // CT_OPENCL_H_          FIXME:
#endif  // _ENABLE_OPENCL_       FIXME:

// EOF