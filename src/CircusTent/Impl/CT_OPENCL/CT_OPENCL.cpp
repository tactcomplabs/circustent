/*
 * _CT_OPENCL_CPP
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_OPENCL.h"
#ifdef _CT_OPENCL_H_

CT_OPENCL::CT_OPENCL(CTBaseImpl::CTBenchType B,
                     CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENCL", B, A),
                                                 Array(nullptr),
                                                 Idx(nullptr),
                                                 memSize(0),
                                                 pes(0),
                                                 iters(0),
                                                 elems(0),
                                                 stride(0),
                                                 numPlatforms(0),
                                                 platformIDs(nullptr),
                                                 targetPlatformID(-1),
                                                 numDevices(0),
                                                 deviceIDs(nullptr),
                                                 targetDeviceID(-1){
}

CT_OPENCL::~CT_OPENCL() {}

void CT_OPENCL::checkOCLError(const char* function, const char* filename, int line, cl_int error){
  if(error != CL_SUCCESS){
    printf("ERROR: %s FAILED! (FILE: %s LINE: %d)\n", function, filename, line);
    printf("Error = %d\n", error);
    fflush(stdout);
    FreeData();
    exit(-1);
  }
}

void CT_OPENCL::printBuildErrors(){
  cl_int error;
  size_t buildLogSize;
  error = clGetProgramBuildInfo(program, deviceIDs[targetDeviceID], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
  checkOCLError("clGetProgramBuildInfo", __FILE__, __LINE__, error);

  char buildLog[buildLogSize];

  error = clGetProgramBuildInfo(program, deviceIDs[targetDeviceID], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL);
  checkOCLError("clGetProgramBuildInfo", __FILE__, __LINE__, error);

  std::cout << buildLog << std::endl;
  return;
}

std::string CT_OPENCL::GetPlatformName(cl_platform_id id){

  cl_int error;
  std::string result;
  size_t size = 0;

  error = clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);
  checkOCLError("clGetPlatformInfo", __FILE__, __LINE__, error);

  result.resize(size);

  error = clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char *>(result.data()), nullptr);
  checkOCLError("clGetPlatformInfo", __FILE__, __LINE__, error);

  return result;
}

std::string CT_OPENCL::GetDeviceName(cl_device_id id){

  cl_int error;
  std::string result;
  size_t size = 0;

  error = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);
  checkOCLError("clGetDeviceInfo", __FILE__, __LINE__, error);

  result.resize(size);

  error = clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char *>(result.data()), nullptr);
  checkOCLError("clGetDeviceInfo", __FILE__, __LINE__, error);

  return result;
}

double CT_OPENCL::Runtime(cl_ulong StartTime, cl_ulong EndTime){
  // Convert nanoseconds to seconds
  return (double) ((EndTime-StartTime) * 1.e-9);
}

// OCL environemnt initialization
bool CT_OPENCL::Initialize(){

  // Ret val for OCL calls
  cl_int error;

  // Check for available platforms
  error = clGetPlatformIDs(0, nullptr, &numPlatforms);
  checkOCLError("clGetPlatformIDs", __FILE__, __LINE__, error);

  if(numPlatforms == 0){
    std::cout << "ERROR: NO OPENCL PLATFORMS FOUND!" << std::endl;
    return false;
  }
  else{
    std::cout << "Found " << numPlatforms << " platforms." << std::endl;
  }

  // Get available platformIDs
  platformIDs = (cl_platform_id*) malloc(sizeof(cl_platform_id)*numPlatforms);
  error = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
  checkOCLError("clGetPlatformIDs", __FILE__, __LINE__, error);

  // Print available platform names/IDs
  std::cout << "Available Platforms:" << std::endl;
  for(cl_uint i = 0; i < numPlatforms; i++){
    std::cout << "Platform ID: " << i << " Platforn Name: " << GetPlatformName(platformIDs[i]) << std::endl;
  }

  // Check that target platform is set
  if(getenv("OCL_TARGET_PLATFORM_NAME") == nullptr){
    std::cout << "ERROR: OCL_TARGET_PLATFORM_NAME NOT SET!" << std::endl;
    return false;
  }

  // Find target platform
  for(cl_uint i = 0; i < numPlatforms; i++){
    if(strcmp(getenv("OCL_TARGET_PLATFORM_NAME"), GetPlatformName(platformIDs[i]).c_str()) == 0){
      std::cout << "Platform ID " << i << " selected." << std::endl;
      targetPlatformID = i;
      break;
    }
  }

  // Ensure the target platform was found
  if(targetPlatformID == -1){
    std::cout << "ERROR: PLATFORM " << std::string(getenv("OCL_TARGET_PLATFORM_NAME")) << " NOT FOUND!" << std::endl;
    return false;
  }

  // Check for available devices on selected platform
  error = clGetDeviceIDs(platformIDs[targetPlatformID], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
  checkOCLError("clGetDeviceIDs", __FILE__, __LINE__, error);

  if(numDevices == 0){
    std::cout << "ERROR: NO OPENCL DEVICES FOUND!" << std::endl;
    return false;
  }
  else{
    std::cout << "Found " << numDevices << " devices." << std::endl;
  }

  // Get available deviceIDs
  deviceIDs = (cl_device_id*) malloc(sizeof(cl_device_id)*numDevices);
  error = clGetDeviceIDs(platformIDs[targetPlatformID], CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, nullptr);
  checkOCLError("clGetDeviceIDs", __FILE__, __LINE__, error);

  // Print available device names/IDs for selected platform
  std::cout << "Available Devices:" << std::endl;
  for(cl_uint i = 0; i < numDevices; i++){
    std::cout << "Device ID: " << i << " Device Name: " << GetDeviceName(deviceIDs[i]) << std::endl;
  }

  // Check that target device is set
  if(getenv("OCL_TARGET_DEVICE_NAME") == nullptr){
    std::cout << "ERROR: OCL_TARGET_DEVICE_NAME NOT SET!" << std::endl;
    return false;
  }

  // Find target device
  for(cl_uint i = 0; i < numDevices; i++){
    if(strcmp(getenv("OCL_TARGET_DEVICE_NAME"), GetDeviceName(deviceIDs[i]).c_str()) == 0){
      std::cout << "Device ID " << i << " selected." << std::endl;
      targetDeviceID = i;
      break;
    }
  }

  // Ensure the target device was found
  if(targetDeviceID == -1){
    std::cout << "ERROR: DEVICE " << std::string(getenv("OCL_TARGET_DEVICE_NAME")) << " NOT FOUND!" << std::endl;
    return false;
  }

  // Create context
  cl_context_properties contextProps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platformIDs[targetPlatformID], 0};
  context = clCreateContext(contextProps, 1, &deviceIDs[targetDeviceID], NULL, NULL, &error);
  checkOCLError("clCreateContext", __FILE__, __LINE__, error);

  // Create command queue
  cl_queue_properties queueProps[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  commandQueue = clCreateCommandQueueWithProperties(context, deviceIDs[targetDeviceID], queueProps, &error);
  checkOCLError("clCreateCommandQueue", __FILE__, __LINE__, error);

  // Set absolute path of kernels source code
  std::string kernelPath(__FILE__);
  kernelPath.replace(kernelPath.find("CT_OPENCL.cpp"), 13, "CT_OPENCL_KERNELS.cl");

  // Create program
  std::ifstream source_stream(kernelPath);
  std::string source_string((std::istreambuf_iterator<char>(source_stream)), std::istreambuf_iterator<char>());
  const char* strings[1] = { source_string.c_str() };
  program = clCreateProgramWithSource(context, 1, strings, NULL, &error);
  checkOCLError("clCreateProgramWithSource", __FILE__, __LINE__, error);

  // Build program, if compile/link errors, print before clean up
  error = clBuildProgram(program, 1, &deviceIDs[targetDeviceID], NULL, NULL, NULL);
  if(error == CL_BUILD_PROGRAM_FAILURE){
    printBuildErrors();
  }
  checkOCLError("clBuildProgram", __FILE__, __LINE__, error);

  return true;
}

bool CT_OPENCL::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType = this->GetAtomType();   // atomic type
  cl_ulong StartTime = 0.;                              // start time
  cl_ulong EndTime = 0.;                                // end time
  double OPS = 0.;                                      // billions of operations
  cl_int error;                                         // OCL ret value
  cl_event kernelComplete;                              // OCL event for kernel execution
  size_t global_pes = (size_t) pes;

  // determine the benchmark type
  if (BType == CT_RAND){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "RAND_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "RAND_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_STRIDE1){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "STRIDE1_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "STRIDE1_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_STRIDEN){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "STRIDEN_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 4, sizeof(cl_ulong), &stride);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "STRIDEN_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 4, sizeof(cl_ulong), &stride);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_PTRCHASE){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "PTRCHASE_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "PTRCHASE_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_SG){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "SG_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(4, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "SG_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(4, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_CENTRAL){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "CENTRAL_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "CENTRAL_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(1, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_SCATTER){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "SCATTER_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "SCATTER_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(3, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else if (BType == CT_GATHER){
    switch (AType){
    case CT_ADD:
      // Create kernel
      kernel = clCreateKernel(program, "GATHER_ADD", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(3, iters, pes);
      break;
    case CT_CAS:
      // Create kernel
      kernel = clCreateKernel(program, "GATHER_CAS", &error);
      checkOCLError("clCreateKernel", __FILE__, __LINE__, error);

      // Add kernel args
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &arrayBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &idxBuffer);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 2, sizeof(cl_ulong), &iters);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);
      clSetKernelArg(kernel, 3, sizeof(cl_ulong), &pes);
      checkOCLError("clSetKernelArg", __FILE__, __LINE__, error);

      // Enqueue kernel execution
      error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                     &global_pes, NULL, 0, NULL, &kernelComplete);
      checkOCLError("clEnqueueNDRangeKernel", __FILE__, __LINE__, error);

      OPS = this->GAM(3, iters, pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }
  else{
    this->ReportBenchError();
    return false;
  }

  // Wait for completion of kernel
  error = clWaitForEvents(1, &kernelComplete);

  // Extract start and end times from profiled event
  error = clGetEventProfilingInfo(kernelComplete, CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &StartTime, NULL);
  checkOCLError("clGetProfilingInfo", __FILE__, __LINE__, error);

  error = clGetEventProfilingInfo(kernelComplete, CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &EndTime, NULL);
  checkOCLError("clGetProfilingInfo", __FILE__, __LINE__, error);
  clReleaseEvent(kernelComplete);

  Timing = this->Runtime(StartTime, EndTime);
  GAMS = OPS/Timing;

  return true;
}

// Allocate Data
bool CT_OPENCL::AllocateData( cl_ulong m,
                              cl_ulong p,
                              cl_ulong i,
                              cl_ulong s){

  // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // Sanity checks
  if (pes == 0){
    std::cout << "CT_OCL::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if (iters == 0){
    std::cout << "CT_OCL::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if (stride == 0){
    std::cout << "CT_OCL::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  cl_ulong end = (pes * iters * stride) - stride;
  if (end >= elems){
    std::cout << "CT_OCL::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  // Allocate memory on host
  Array = (cl_ulong *)(malloc(memSize));
  if (Array == nullptr){
    std::cout << "CT_OCL::AllocateData : 'Array' could not be allocated" << std::endl;
    return false;
  }

  Idx = (cl_ulong *)(malloc(sizeof(cl_ulong) * (pes + 1) * iters));
  if (Idx == nullptr){
    std::cout << "CT_OCL::AllocateData : 'Idx' could not be allocated" << std::endl;
    free(Array);
    return false;
  }

  // init the arrays
  srand(time(NULL));
  if (this->GetBenchType() == CT_PTRCHASE){
    for (unsigned i = 0; i < ((pes + 1) * iters); i++){
      Idx[i] = (cl_ulong)(rand() % ((pes + 1) * iters));
    }
  }
  else{
    for (cl_ulong i = 0; i < ((pes + 1) * iters); i++){
      Idx[i] = (cl_ulong)(rand() % (elems - 1));
    }
  }
  for (unsigned i = 0; i < elems; i++){
    Array[i] = (cl_ulong)(rand());
  }

  cl_int error;
  cl_event arrayWrite, idxWrite;

  // Create OpenCL buffers using initialized arrays
  arrayBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                               memSize, Array, &error);
  checkOCLError("clCreateBuffer", __FILE__, __LINE__, error);
  idxBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                             (sizeof(cl_ulong) * (pes + 1) * iters), Idx, &error);
  checkOCLError("clCreateBuffer", __FILE__, __LINE__, error);

  // Transfer to target device and wait for completion
  error = clEnqueueWriteBuffer(commandQueue, arrayBuffer, CL_FALSE, 0, memSize,
                               Array, 0, NULL, &arrayWrite);
  checkOCLError("clEnqueueWriteBuffer", __FILE__, __LINE__, error);
  error = clWaitForEvents(1, &arrayWrite);
  checkOCLError("clWaitForEvents", __FILE__, __LINE__, error);
  clReleaseEvent(arrayWrite);

  error = clEnqueueWriteBuffer(commandQueue, idxBuffer, CL_FALSE, 0, (sizeof(cl_ulong) * (pes + 1) * iters),
                               Idx, 0, NULL, &idxWrite);
  checkOCLError("clEnqueueWriteBuffer", __FILE__, __LINE__, error);
  error = clWaitForEvents(1, &idxWrite);
  checkOCLError("clWaitForEvents", __FILE__, __LINE__, error);
  clReleaseEvent(idxWrite);

  // Print PEs configuration
  std::cout << "SETTING NUM_PEs = " << pes << std::endl;

  return true;
}

//Free data
bool CT_OPENCL::FreeData(){

  // First release OCL resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(arrayBuffer);
  clReleaseMemObject(idxBuffer);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);

  // Release memory
  if (Array) {
    free(Array);
  }
  if (Idx) {
    free(Idx);
  }
  if (platformIDs) {
    free(platformIDs);
  }
  if (deviceIDs) {
    free(deviceIDs);
  }

  return true;
}
#endif

// ==============================================================
// EOF
