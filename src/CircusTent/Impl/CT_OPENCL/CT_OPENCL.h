/*
 * CT_OPENCL_H
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

#ifdef _ENABLE_OPENCL_

#ifndef _CT_OPENCL_H_
#define _CT_OPENCL_H_

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl2.hpp>
#endif

#include "CircusTent/CTBaseImpl.h"

class CT_OPENCL : public CTBaseImpl{
private:
  cl_ulong *Array;              ///< CT_OPENCL: Data array
  cl_mem arrayBuffer;           ///< CT_OPENCL: OCL array buffer object
  cl_ulong *Idx;                ///< CT_OPENCL: Index array
  cl_mem idxBuffer;             ///< CT_OPENCL: OCL idx array buffer object
  cl_ulong memSize;             ///< CT_OPENCL: Memory size (in bytes)
  cl_ulong pes;                 ///< CT_OPENCL: Number of processing elements
  cl_ulong iters;               ///< CT_OPENCL: Number of iterations per pe
  cl_ulong elems;               ///< CT_OPENCL: Number of u8 elements
  cl_ulong stride;              ///< CT_OPENCL: Stride in elements
  cl_uint numPlatforms;         ///< CT_OPENCL: Number of detected platforms
  cl_platform_id *platformIDs;  ///< CT_OPENCL: Array of platform IDs
  int targetPlatformID;         ///< CT_OPENCL: Index of target platform
  cl_uint numDevices;           ///< CT_OPENCL: Number of detected devices for selected platform
  cl_device_id *deviceIDs;      ///< CT_OPENCL: Array of device IDs for selected platform
  int targetDeviceID;           ///< CT_OPENCL: Index of target device for selected platform
  cl_context context;           ///< CT_OPENCL: OCL context
  cl_command_queue commandQueue;///< CT_OPENCL: OCL command queue
  cl_program program;           ///< CT_OPENCL: OCL program containing CircusTent kernels
  cl_kernel kernel;             ///< CT_OPENCL: Selected kernel for execution

public:
  /// CircusTent OpenCL constructor
  CT_OPENCL(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent OpenCL destructor
  ~CT_OPENCL();

  // Helper functions
  void checkOCLError(const char* function, const char* filename, int line, cl_int error);
  void printBuildErrors();
  std::string GetPlatformName(cl_platform_id id);
  std::string GetDeviceName(cl_device_id id);

  // Run time calculation for OpenCL implementaion, overrides CT_BASE
  double Runtime(cl_ulong StartTime, cl_ulong EndTime);

  // CircusTent OpenCL initialization function
  bool Initialize();

  /// CircusTent OpenCL execution function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent OpenCL data allocation function
  virtual bool AllocateData( cl_ulong memSize,
                             cl_ulong pes,
                             cl_ulong iters,
                             cl_ulong stride ) override;

  /// CircusTent OpenCL data free function
  virtual bool FreeData() override;
};

#endif  // CT_OPENCL_H_
#endif  // _ENABLE_OPENCL_

// ==============================================================
// EOF
