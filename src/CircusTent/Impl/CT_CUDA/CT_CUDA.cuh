/*
 * FIXME: ensure proper file extension is used throughout the head of this file.
 * CT_CUDA_CUH
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

// FIXME: check include path to CUDA runtime
#include <cuda_runtime.h>
// #include "/usr/local/cuda-11.0/include/cuda_runtime.h" // if cuda_runtime.h doesn't work on HPCC 
#include <cuda.h>


#include "CircusTent/CTBaseImpl.h"

// TODO: Benchmark Kernel Prototypes

class CT_CUDA : public CTBaseImpl {
private:
  // FIXME: I think we need device copies of some of these class attributes
  uint64_t *Array;             ///< CT_CUDA: Host data array
  uint64_t *Idx;               ///< CT_CUDA: Host index array

  uint64_t *d_Array;           ///< CT_CUDA: Device copy of data array
  uint64_t *d_Idx;             ///< CT_CUDA: Device copy of index array

  uint64_t memSize;            ///< CT_CUDA: Memory size (in bytes)
  uint64_t pes;                ///< CT_CUDA: Number of processing elements
  uint64_t iters;              ///< CT_CUDA: Number of iterations per thread
  uint64_t elems;              ///< CT_CUDA: Number of u8 elements
  uint64_t stride;             ///< CT_CUDA: Stride in elements

  // cudaDeviceProp prop;         ///< CT_CUDA: Variable that hold CUDA device properties
  int deviceID;                ///< CT_CUDA: CUDA device ID
  // int deviceCount;             ///< CT_CUDA: Number of usable CUDA devices

  // TODO: set these vars via command line options
  int blocksPerGrid;        ///< CT_CUDA: Number of blocks per Grid
  int threadsPerBlock;      ///< CT_CUDA: Number of threads per block


public:
  // CircusTent CUDA constructor
  CT_CUDA(CTBaseImpl::CTBenchType B,
    CTBaseImpl::CTAtomType A);

  // CircusTent CUDA destructor
  ~CT_CUDA();

  // TODO: Helper functions
  bool parseCUDAOpts(int argc, char **argv);
  void printDeviceProperties(int deviceID);

  // CircusTent CUDA execution function
  virtual bool Execute(double &Timing, double &GAMS) override;

  // CircusTent CUDA data allocation function
  // TODO: probably need to copy arrays to device here
  virtual bool AllocateData( uint64_t memSize,
    uint64_t pes,
    uint64_t iters,
    uint64_t stride ) override;

  // CircusTent CUDA data free function
  virtual bool FreeData() override;
}

#endif  // CT_CUDA_CUH_
#endif  // _ENABLE_CUDA_