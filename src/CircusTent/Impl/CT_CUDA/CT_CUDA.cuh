/*
 * _CT_CUDA_CUH_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

 /**
  * \class CT_CUDA
  *
  * \ingroup CircusTent
  *
  * \brief CircusTent CUDA Implementation
  **/

#ifdef _ENABLE_CUDA_

#ifndef _CT_CUDA_CUH_
#define _CT_CUDA_CUH_

// #include <cuda_runtime.h>
#include "/usr/local/cuda-11.0/include/cuda_runtime.h"
#include "CircusTent/CTBaseImpl.h"

// Benchmark Kernel Prototypes
extern "C" {
  __global__ void RAND_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void RAND_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void STRIDE1_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters
  );

  __global__ void STRIDE1_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters
  );

  __global__ void STRIDEN_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters,
    uint64_t stride
  );

  __global__ void STRIDEN_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters,
    uint64_t stride
  );

  __global__ void CENTRAL_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters
  );

  __global__ void CENTRAL_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t iters
  );

  __global__ void PTRCHASE_ADD(
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void PTRCHASE_CAS(
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void SG_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void SG_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void SCATTER_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void SCATTER_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void GATHER_ADD(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );

  __global__ void GATHER_CAS(
    uint64_t* __restrict__ ARRAY,
    uint64_t* __restrict__ IDX,
    uint64_t iters
  );
}

class CT_CUDA : public CTBaseImpl {
private:
  uint64_t *Array;             ///< CT_CUDA: Host data array
  uint64_t *Idx;               ///< CT_CUDA: Host index array

  uint64_t *d_Array;           ///< CT_CUDA: Device copy of data array
  uint64_t *d_Idx;             ///< CT_CUDA: Device copy of index array

  uint64_t memSize;            ///< CT_CUDA: Memory size (in bytes)
  uint64_t iters;              ///< CT_CUDA: Number of iterations per thread block
  uint64_t elems;              ///< CT_CUDA: Number of u8 elements
  uint64_t stride;             ///< CT_CUDA: Stride in elements
  
  uint64_t threadBlocks;       ///< CT_CUDA: Number of threadBlocks
  uint64_t threadsPerBlock;    ///< CT_CUDA: Number of threads per thread block

public:
  // CircusTent CUDA constructor
  CT_CUDA(CTBaseImpl::CTBenchType B,
          CTBaseImpl::CTAtomType A);

  // CircusTent CUDA destructor
  ~CT_CUDA();

  // CircusTent CUDA data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t blocks,
                             uint64_t threads,
                             uint64_t iters,
                             uint64_t stride ) override;

  // CircusTent CUDA execution function
  virtual bool Execute(double &Timing, double &GAMS) override;

  // CircusTent CUDA data free function
  virtual bool FreeData() override;

  // Helper functions
  bool PrintCUDADeviceProperties();
};

#endif  // _CT_CUDA_CUH_
#endif  // _ENABLE_CUDA_

// EOF
