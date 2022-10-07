//
// _CT_OPENACC_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_OPENACC.h"

#ifdef _CT_OPENACC_H_

CT_OPENACC::CT_OPENACC(CTBaseImpl::CTBenchType B,
                CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENACC",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0),
                                           deviceTypeStr(""),
                                           deviceID(-1) {
}

CT_OPENACC::~CT_OPENACC(){
}

bool CT_OPENACC::Execute(double &Timing, double &GAMS){

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

bool CT_OPENACC::AllocateData( uint64_t m,
                               uint64_t p,
                               uint64_t i,
                               uint64_t s){
  // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // check args
  if( pes == 0 ){
    std::cout << "CT_OPENACC::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_OPENACC::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_OPENACC::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (pes * iters * stride);
  if( end > elems ){
    std::cout << "CT_OPENACC::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  // data on the target device
  Array = (uint64_t *) acc_malloc(memSize);
  // temporary data array for initialization on the host
  uint64_t *HostArray = (uint64_t *) malloc(memSize);
  if( ( Array == nullptr ) || ( HostArray == nullptr ) ){
    std::cout << "CT_OPENACC::AllocateData : 'Array' could not be allocated" << std::endl;
    acc_free(Array);
    free(HostArray);
    return false;
  }

  // target and host Idx arrays
  Idx = (uint64_t *) acc_malloc(sizeof(uint64_t)*(pes+1)*iters);
  uint64_t *HostIdx = (uint64_t *) malloc(sizeof(uint64_t)*(pes+1)*iters);
  if( ( Idx == nullptr ) || ( HostIdx == nullptr ) ){
    std::cout << "CT_OPENACC::AllocateData : 'Idx' could not be allocated" << std::endl;
    acc_free(Array);
    free(HostArray);
    acc_free(Idx);
    free(HostIdx);
    return false;
  }

  // Randomize arrays on the host
  srand(time(NULL));
  if( this->GetBenchType() == CT_PTRCHASE ){
    for( unsigned i=0; i<((pes+1)*iters); i++ ){
      HostIdx[i] = (uint64_t)(rand()%((pes+1)*iters));
    }
  }else{
    for( unsigned i=0; i<((pes+1)*iters); i++ ){
      HostIdx[i] = (uint64_t)(rand()%(elems-1));
    }
  }
  for( unsigned i=0; i<elems; i++ ){
    HostArray[i] = (uint64_t)(rand());
  }

  // Copy initalized arrays to target
  acc_memcpy_to_device(Array, HostArray, memSize);
  acc_memcpy_to_device(Idx, HostIdx, sizeof(uint64_t)*(pes+1)*iters);

  // Free temp arrays on host
  free(HostArray);
  free(HostIdx);

  printf("RUNNING WITH NUM_GANGS = %lu\n", pes);


  return true;
}

bool CT_OPENACC::SetDevice(){

  // Check that target device type is set in the environment
  if(getenv("ACC_DEVICE_TYPE") == nullptr){
      std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_TYPE is not set!" << std::endl;
      return false;
  }

  deviceTypeStr.assign(getenv("ACC_DEVICE_TYPE"));
  std::cout << "Target device type set to " << deviceTypeStr << std::endl;

  // Check that target devices of this type are found
  deviceTypeEnum = acc_get_device_type();
  int numDevs = acc_get_num_devices(deviceTypeEnum);
  if(!numDevs){
      std::cout << "CT_OPENACC::SetDevice : Unable to locate target devices of type "
                << deviceTypeStr << std::endl;
      return false;
  }

  // Check that target device ID is set in the environment
  if(getenv("ACC_DEVICE_NUM") == nullptr){
      std::cout << "CT_OPENACC::SetDevice : ACC_DEVICE_NUM is not set!" << std::endl;
      return false;
  }

  deviceID = atoi(getenv("ACC_DEVICE_NUM"));

  // Check that we are set to run on target device ID
  if(acc_get_device_num(deviceTypeEnum) != deviceID){
      std::cout << "CT_OPENACC::SetDevice : Unable to set target ID " \
                << deviceID << std::endl;
      return false;
  }

  // Print target ID
  std::cout << "Target device ID set to " << deviceID \
            << " of " << acc_get_num_devices(deviceTypeEnum) \
            << " total devices" << std::endl;

  // Init OpenACC
  acc_init(deviceTypeEnum);

  return true;
}

bool CT_OPENACC::FreeData(){
  if( Array ){
    acc_free(Array);
  }
  if( Idx ){
    acc_free(Idx);
  }

  // Close OpenACC
  acc_shutdown(deviceTypeEnum);

  return true;
}

#endif

// EOF
