//
// _CT_OMP_TARGET_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_OMP_TARGET.h"

#ifdef _CT_OMP_TARGET_H_

CT_OMP_TARGET::CT_OMP_TARGET(CTBaseImpl::CTBenchType B,
                CTBaseImpl::CTAtomType A) : CTBaseImpl("OMP_TARGET",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0),
                                           deviceID(-1) {
}

CT_OMP_TARGET::~CT_OMP_TARGET(){
}

bool CT_OMP_TARGET::Execute(double &Timing, double &GAMS){

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
    }
  }else{
    this->ReportBenchError();
    return false;
  }

  Timing = this->Runtime(StartTime,EndTime);
  GAMS   = OPS/Timing;

  return true;
}

bool CT_OMP_TARGET::AllocateData( uint64_t m,
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
    std::cout << "CT_OMP_TARGET::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_OMP_TARGET::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_OMP_TARGET::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (pes * iters * stride) - stride;
  if( end >= elems ){
    std::cout << "CT_OMP_TARGET::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }


  // allocate the data on the target device
  Array = (uint64_t *) omp_target_alloc(memSize, deviceID);
  // temporary data array for initialization on the host
  uint64_t *HostArray = (uint64_t *) malloc(memSize);

  if( ( Array == nullptr ) || ( HostArray == nullptr ) ){
    std::cout << "CT_OMP_TARGET::AllocateData : 'Array' could not be allocated" << std::endl;
    std::cout << "Array = " << Array << " HostArray = " << HostArray << std::endl;
    omp_target_free(Array, deviceID);
    free(HostArray);
    return false;
  }

  // target and host Idx arrays
  Idx = (uint64_t *) omp_target_alloc(sizeof(uint64_t)*(pes+1)*iters, deviceID);
  uint64_t *HostIdx = (uint64_t *) malloc(sizeof(uint64_t)*(pes+1)*iters);

  if( ( Idx == nullptr ) || ( HostIdx == nullptr ) ){
    std::cout << "CT_OMP_TARGET::AllocateData : 'Idx' could not be allocated" << std::endl;
    omp_target_free(Array, deviceID);
    omp_target_free(Idx, deviceID);
    free(HostArray);
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
  omp_target_memcpy(Array, HostArray, memSize, 0, 0, deviceID, omp_get_initial_device());
  omp_target_memcpy(Idx, HostIdx, sizeof(uint64_t)*(pes+1)*iters, 0, 0, deviceID, omp_get_initial_device());

  // Free temp arrays on host
  free(HostArray);
  free(HostIdx);

  // Set the number of teams on device
  // Need OpenMP 5.1 support, for now using num_teams clause at kernel directives
  //omp_set_num_teams(pes);

  // Sanity check on target
  #pragma omp target teams num_teams(pes)
  {
    if(omp_get_team_num() == 0){
        printf("RUNNING WITH NUM_TEAMS = %d\n", omp_get_num_teams());
    }
  }

  return true;
}

bool CT_OMP_TARGET::SetDevice(){

  // Check that target devices are detected
  if(!omp_get_num_devices()){
    std::cout << "CT_OMP_TARGET::SetDevice : No target devices detected!" << std::endl;
    return false;
  }

  // Check if target device ID is overridden in the environment
  if(getenv("OMP_DEFAULT_DEVICE") == nullptr){
    std::cout << "CT_OMP_TARGET::SetDevice : OMP_DEFAULT_DEVICE is not set." << std::endl;
  }

  deviceID = omp_get_default_device();
  std::cout << "CT_OMP_TARGET::SetDevice : Target device ID = " << deviceID << " of " \
            << omp_get_num_devices() << " total detected devices." << std::endl;

  return true;
}

bool CT_OMP_TARGET::FreeData(){
  if( Array ){
    omp_target_free(Array, deviceID);
  }
  if( Idx ){
    omp_target_free(Idx, deviceID);
  }
  return true;
}

#endif

// EOF
