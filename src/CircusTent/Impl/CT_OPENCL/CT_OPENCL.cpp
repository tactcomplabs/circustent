/*
 * TODO: _CT_OPENCL_IMPL_C
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
using namespace cl;

CT_OPENCL::CT_OPENCL(CTBaseImpl::CTBenchType B,
                     CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENCL", B, A),
                                                 Array(nullptr),
                                                 Idx(nullptr),
                                                 memSize(0),
                                                 pes(0),
                                                 iters(0),
                                                 elems(0),
                                                 stride(0),
                                                 deviceTypeStr(""), // FIXME:
                                                 deviceID(-1)       // FIXME:
{
}

CT_OPENCL::~CT_OPENCL(){}

bool CT_OPENCL::Execute(double &Timing, double &GAMS)
{
  // FIXME:
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
      RAND_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      RAND_CAS(Array, Idx, iters, pes);
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
      STRIDE1_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      STRIDE1_CAS(Array, Idx, iters, pes);
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
      STRIDEN_ADD(Array, Idx, iters, pes, stride);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      STRIDEN_CAS(Array, Idx, iters, pes, stride);
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
      PTRCHASE_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      PTRCHASE_CAS(Array, Idx, iters, pes);
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
      SG_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(4, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      SG_CAS(Array, Idx, iters, pes);
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
      CENTRAL_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      CENTRAL_CAS(Array, Idx, iters, pes);
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
      SCATTER_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      SCATTER_CAS(Array, Idx, iters, pes);
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
      GATHER_ADD(Array, Idx, iters, pes);
      EndTime = this->MySecond();
      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      GATHER_CAS(Array, Idx, iters, pes);
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
      std::cout << "RUNNING WITH NUM_THREADS = " << get_global_size(0) << std::endl; // FIXME:
    }
  }

  return true;
}

// ---------------------------------------------------------
// TODO:
bool CT_OPENCL::FreeData()
{

}
// ---------------------------------------------------------


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


// ************ THIS PART ONLY NEEDS TO BE DONE WHEN SPECIFIED IN "circustent" comamand ARGS ***************
// Create a kernel from the program
// TODO: does this need to be done for each kernel?
cl_kernel RAND_ADD = clCreateKernel(program, "RAND_ADD", NULL);

// FIXME: Specify arguments to the kernel
// TODO: does this need to be done for each kernel?
// TODO: Pass in the arguments given in the command line
  clSetKernelArg(RAND_ADD, 0, sizeof(cl_mem), &d_a);
  clSetKernelArg(RAND_ADD, 1, sizeof(cl_mem), &d_b);
  clSetKernelArg(RAND_ADD, 2, sizeof(cl_mem), &d_c);
  clSetKernelArg(RAND_ADD, 3, sizeof(unsigned int), &n);

// Run the kernel
// TODO: does this need to be done for each kernel
clEnqueueNDRangeKernel(queue, RAND_ADD, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

#endif

// ==============================================================
// EOF