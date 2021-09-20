/*
 * _CT_OPENCL_IMPL_C
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#ifdef _CT_OPENCL_H_
#include "CT_OPENCL.h"
#include <fstream>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace cl;

// ************* OpenCL Setup Code ***************
// Getting OpenCL Platform
std::vector<cl_platform_id> GetPlatforms() {
    cl_uint platformIdCount = 0;
  clGetPlatformIDs(0, NULL, &platformIdCount);

  if (platformIdCount == 0) {
    std::cerr << "No OpenCL platform found" << std::endl;
    exit(1);
  } else {
    std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
  }
  std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);
    return platformIds;
}

// Get the devices for the OpenCL Platform
std::vector<cl_device_id> GetDevices(cl_platform_id platform) {
    cl_uint deviceIdCount = 0;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);

  if (deviceIdCount == 0) {
    std::cerr << "No OpenCL devices found" << std::endl;
    exit(1);
  } else {
    std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
  }

  std::vector<cl_device_id> deviceIds(deviceIdCount);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);
    return deviceIds;
}

// Create an OpenCL context for a specified device
cl_context context = clCreateContext(0, 1, &deviceIds[device_num], NULL, NULL, NULL);

// Create a Command Queue (with profiling enabled, needed for timing kernels)
cl_command_queue queue = clCreateCommandQueue(context, deviceIds[device_num], CL_QUEUE_PROFILING_ENABLE, NULL);

// Creare program for specified context
std::string LoadKernel(const char* name) {
    std::ifstream in(name);
  std::string result((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return result;
}
cl_program CreateProgram(const std::string& source, cl_context context) {
    size_t lengths[1] = { source.size() };
  const char* sources[1] = { source.data() };
  cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, NULL);
  return program;
}
cl_program program = CreateProgram(LoadKernel("CT_OPENCL_KERNELS.cl"), context);

// Build the program
clBuildProgram(program, 0, NULL, "-cl-mad-enable", NULL, NULL);

// TODO: Do I need to create a buffer here?

// ************* END OF OpenCL Setup Code ***************


CT_OPENCL::CT_OPENCL(CTBaseImpl::CTBenchType B,
                     CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENCL", B, A),
                                                 Array(nullptr),
                                                 Idx(nullptr),
                                                 memSize(0),
                                                 pes(0),
                                                 iters(0),
                                                 elems(0),
                                                 stride(0),
{
}

CT_OPENCL::~CT_OPENCL(){}

bool CT_OPENCL::Execute(double &Timing, double &GAMS)
{
  CTBaseImpl::CTBenchType BType = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType = this->GetAtomType();   // atomic type
  double StartTime = 0.;                                // start time
  double EndTime = 0.;                                  // end time
  double OPS = 0.;                                      // billions of operations

  // determine the benchmark type
  if (BType == CT_RAND)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel RAND_ADD = clCreateKernel(program, "RAND_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(RAND_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(RAND_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(RAND_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(RAND_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, RAND_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel RAND_CAS = clCreateKernel(program, "RAND_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(RAND_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(RAND_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(RAND_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(RAND_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, RAND_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_STRIDE1)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel STRIDE1_ADD = clCreateKernel(program, "STRIDE1_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(STRIDE1_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(STRIDE1_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(STRIDE1_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(STRIDE1_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, STRIDE1_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel STRIDE1_CAS = clCreateKernel(program, "STRIDE1_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(STRIDE1_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(STRIDE1_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(STRIDE1_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(STRIDE1_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, STRIDE1_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_STRIDEN)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel STRIDEN_ADD = clCreateKernel(program, "STRIDEN_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(STRIDEN_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(STRIDEN_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(STRIDEN_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(STRIDEN_ADD, 3, sizeof(cl_mem), pes);
      clSetKernelArg(STRIDEN_ADD, 4, sizeof(cl_mem), stride);
      clEnqueueNDRangeKernel(queue, STRIDEN_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel STRIDEN_CAS = clCreateKernel(program, "STRIDEN_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(STRIDEN_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(STRIDEN_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(STRIDEN_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(STRIDEN_CAS, 3, sizeof(cl_mem), pes);
      clSetKernelArg(STRIDEN_CAS, 4, sizeof(cl_mem), stride);
      clEnqueueNDRangeKernel(queue, STRIDEN_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_PTRCHASE)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel PTRCHASE_ADD = clCreateKernel(program, "PTRCHASE_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(PTRCHASE_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(PTRCHASE_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(PTRCHASE_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(PTRCHASE_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, PTRCHASE_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel PTRCHASE_CAS = clCreateKernel(program, "PTRCHASE_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(PTRCHASE_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(PTRCHASE_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(PTRCHASE_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(PTRCHASE_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, PTRCHASE_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_SG)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel SG_ADD = clCreateKernel(program, "SG_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(SG_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(SG_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(SG_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(SG_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, SG_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(4, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel SG_CAS = clCreateKernel(program, "SG_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(SG_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(SG_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(SG_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(SG_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, SG_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(4, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_CENTRAL)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel CENTRAL_ADD = clCreateKernel(program, "CENTRAL_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(CENTRAL_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(CENTRAL_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(CENTRAL_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(CENTRAL_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, CENTRAL_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel CENTRAL_CAS = clCreateKernel(program, "CENTRAL_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(CENTRAL_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(CENTRAL_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(CENTRAL_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(CENTRAL_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, CENTRAL_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_SCATTER)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel SCATTER_ADD = clCreateKernel(program, "SCATTER_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(SCATTER_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(SCATTER_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(SCATTER_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(SCATTER_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, SCATTER_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel SCATTER_CAS = clCreateKernel(program, "SCATTER_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(SCATTER_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(SCATTER_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(SCATTER_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(SCATTER_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, SCATTER_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_GATHER)
  {
    switch (AType)
    {
    case CT_ADD:
      StartTime = this->MySecond();
      cl_kernel GATHER_ADD = clCreateKernel(program, "GATHER_ADD", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(GATHER_ADD, 0, sizeof(cl_mem), Array);
      clSetKernelArg(GATHER_ADD, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(GATHER_ADD, 2, sizeof(cl_mem), iters);
      clSetKernelArg(GATHER_ADD, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, GATHER_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      cl_kernel GATHER_CAS = clCreateKernel(program, "GATHER_CAS", NULL);
       // XXX: arg might need to be a pointer
      clSetKernelArg(GATHER_CAS, 0, sizeof(cl_mem), Array);
      clSetKernelArg(GATHER_CAS, 1, sizeof(cl_mem), Idx);
      clSetKernelArg(GATHER_CAS, 2, sizeof(cl_mem), iters);
      clSetKernelArg(GATHER_CAS, 3, sizeof(cl_mem), pes);
      clEnqueueNDRangeKernel(queue, GATHER_CAS, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else
  {
    this->ReportBenchError();
    return false;
  }

  Timing = this->Runtime(StartTime, EndTime);
  GAMS = OPS / Timing;

  return true;
}

bool CT_OPENCL::AllocateData(
    uint64_t m,
    uint64_t p,
    uint64_t i,
    uint64_t s)
{
  // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // allocate all the memory
  if (pes == 0)
  {
    std::cout << "CT_OCL::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if (iters == 0)
  {
    std::cout << "CT_OCL::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if (stride == 0)
  {
    std::cout << "CT_OCL::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize / 8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (pes * iters * stride) - stride;
  if (end > elems)
  {
    std::cout << "CT_OCL::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  Array = (uint64_t *)(malloc(memSize));
  if (Array == nullptr)
  {
    std::cout << "CT_OCL::AllocateData : 'Array' could not be allocated" << std::endl;
    return false;
  }

  Idx = (uint64_t *)(malloc(sizeof(uint64_t) * (pes + 1) * iters));
  if (Idx == nullptr)
  {
    std::cout << "CT_OCL::AllocateData : 'Idx' could not be allocated" << std::endl;
    free(Array);
    return false;
  }

  // initiate the random array
  srand(time(NULL));
  if (this->GetBenchType() == CT_PTRCHASE)
  {
    for (unsigned i = 0; i < ((pes + 1) * iters); i++)
    {
      Idx[i] = (uint64_t)(rand() % ((pes + 1) * iters));
    }
  }
  else
  {
    for (unsigned i = 0; i < ((pes + 1) * iters); i++)
    {
      Idx[i] = (uint64_t)(rand() % (elems - 1));
    }
  }
  for (unsigned i = 0; i < elems; i++)
  {
    Array[i] = (uint64_t)(rand());
  }

#pragma ocl parallel
  {
#pragma ocl single
    {
      std::cout << "RUNNING WITH NUM_THREADS = " << get_global_size(0) << std::endl;
    }
  }

  return true;
}

// ---------------------------------------------------------
bool CT_OPENCL::FreeData()
{
  // TODO:
  if (Array) {

  }
  if (Idx) {

  }

  // Close OpenCL

  return true;
}

#endif

// ==============================================================
// EOF