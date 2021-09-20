/*
 * CT_OPENCL__H
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
#include <ctime>

// -------------------------
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
// -------------------------

#include "CircusTent/CTBaseImpl.h"

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
};

#endif  // CT_OPENCL_H_          FIXME:
#endif  // _ENABLE_OPENCL_       FIXME:

// ==============================================================
// EOF