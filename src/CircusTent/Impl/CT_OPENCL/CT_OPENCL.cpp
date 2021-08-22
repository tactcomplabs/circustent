/*
 * TODO: _CT_OPENCL_TARGET_IMPL_C
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
                CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENCL",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0),
                                           deviceTypeStr(""), // FIXME:
                                           deviceID(-1) {} // FIXME:

CT_OPENCL::~CT_OPENCL() {

}

bool CT_OPENCL::Execute(double &Timing, double &GAMS) {
    // --------------------------------
    // FIXME:
    // --------------------------------
    CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type
  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations

  // determine the benchmark type
  if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      RAND_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      RAND_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_STRIDE1 ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      STRIDE1_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      STRIDE1_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_STRIDEN ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      STRIDEN_ADD( Array, Idx, iters, pes, stride );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      STRIDEN_CAS( Array, Idx, iters, pes, stride );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_PTRCHASE ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      PTRCHASE_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      PTRCHASE_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_SG ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      SG_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      SG_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_CENTRAL ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      CENTRAL_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      CENTRAL_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_SCATTER ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      SCATTER_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      SCATTER_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else if( BType == CT_GATHER ){
    switch( AType ){
    case CT_ADD:
      StartTime = this->MySecond();
      GATHER_ADD( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      StartTime = this->MySecond();
      GATHER_CAS( Array, Idx, iters, pes );
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
      break;
    }
  }else{
    this->ReportBenchError();
    return false;
  }

  Timing = this->Runtime(StartTime,EndTime);
  GAMS   = OPS/Timing;

  return true;
}

bool CT_OPENCL::AllocateData(
    uint64_t m,
    uint64_t p,
    uint64_t i,
    uint64_t s
) {
    // FIXME:
    // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // allocate all the memory
  if( pes == 0 ){
    std::cout << "CT_OCL::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_OCL::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_OCL::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (pes * iters * stride)-stride;
  if( end > elems ){
    std::cout << "CT_OCL::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  Array = (uint64_t *)(malloc( memSize ));
  if( Array == nullptr ){
    std::cout << "CT_OCL::AllocateData : 'Array' could not be allocated" << std::endl;
    return false;
  }

  Idx = (uint64_t *)(malloc( sizeof(uint64_t) * (pes+1) * iters ));
  if( Idx == nullptr ){
    std::cout << "CT_OCL::AllocateData : 'Idx' could not be allocated" << std::endl;
    free( Array );
    return false;
  }

  // initiate the random array
  srand(time(NULL));
  if( this->GetBenchType() == CT_PTRCHASE ){
    for( unsigned i=0; i<((pes+1)*iters); i++ ){
      Idx[i] = (uint64_t)(rand()%((pes+1)*iters));
    }
  }else{
    for( unsigned i=0; i<((pes+1)*iters); i++ ){
      Idx[i] = (uint64_t)(rand()%(elems-1));
    }
  }
  for( unsigned i=0; i<elems; i++ ){
    Array[i] = (uint64_t)(rand());
  }

  // TODO: init the OpenCL context
  // FIXME: omp_set_num_threads(pes);

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
// FIXME: this might be completely wrong
// FIXME: copied from: https://www.cl.cam.ac.uk/teaching/1819/AdvGraphIP/03_OpenCL.pdf
bool OPENCL::SetDevice() {
    // todo
    // get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
      std::cout << "No platforms found. Check OpenCL installation!\n"
      exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n"

    // get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
      std::cout << "No devices found. Check OpenCL installation!\n";
      exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
}
// ---------------------------------------------------------
// FIXME:
bool CT_OPENCL::FreeData() {
    // todo
    if( Array ){
      free( Array );
    }
    if( Idx ){
      free( Idx );
    }
    return true;
}
// ---------------------------------------------------------

#endif

// ==============================================================
// EOF