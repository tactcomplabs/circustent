//
// _CT_YGM_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_YGM.h"

#ifdef _CT_YGM_H_

CT_YGM::CT_YGM(CTBaseImpl::CTBenchType B,
               CTBaseImpl::CTAtomType A) : CTBaseImpl("YGM",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           Target(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0) {
}

CT_YGM::~CT_YGM(){}

bool CT_YGM::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type

  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations
  int rank          = -1; // mpi rank

  if( world.rank() == 0 ){
    std::cout << "Beginning test execution" << std::endl;
  }

  if( BType == CT_STRIDE1 ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      STRIDE1_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      STRIDE1_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_STRIDEN ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      STRIDEN_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      STRIDEN_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_CENTRAL ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      CENTRAL_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      CENTRAL_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else{
    this->ReportBenchError();
    return false;
  }

  Timing = this->Runtime(StartTime,EndTime);
  GAMS   = OPS/Timing;

  return true;
}

bool CT_YGM::AllocateData( uint64_t m,
                           uint64_t p,
                           uint64_t i,
                           uint64_t s){

  int rank          = -1; // ygm rank
  int size          = -1; // ygm size (num pe's)

  // save the benchmark setup data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // allocate all the memory
  if( pes == 0 ){
    std::cout << "CT_YGM::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_YGM::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_YGM::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  elems = (memSize/8);

  uint64_t end = (iters * stride)-stride;
  if( end >= elems ){
    std::cout << "CT_YGM::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << "; stride =" << stride
              << std::endl;
    return false;
  }

  if( !(this->GetBenchType() == CT_PTRCHASE) ){
    if( elems < iters ){
      std::cout << "CT_YGM::AllocateData : Memory size is too small for iteration count" << std::endl;
      std::cout << "                       : Increase the memory footprint per PE or reduce the iteration count" << std::endl;
      return false;
    }
  }

  rank = world.rank();
  size = world.size();
  world.barrier();

  // 'Array' resides in local heap space
  Array = new std::uint64_t[elems];
  if ( Array == nullptr ){
    world.cout("CT_YGM::AllocateData : 'Array' could not be allocated");
    return false;
  }

  // Create ygm ptr to local Array
  // make_ygm_ptr will check ready to use
  yp_Array = world.make_ygm_ptr(Array);

  // 'Idx' resides in local heap space
  Idx = new std::uint64_t[iters + 1];

  if ( Idx == nullptr ){
    world.cout("CT_YGM::AllocateData : 'Idx' could not be allocated");
    delete[] Array;
    return false;
  }

  // Create ygm ptr to local Idx
  // make_ygm_ptr will check ready to use
  yp_Idx = world.make_ygm_ptr(Idx);

  // 'Target' is local to PE
  Target = new int[iters];

  if (Target == nullptr)
  {
    std::cout << "CT_YGM:AllocateData: 'Target' could not be allocated" << std::endl;
    delete[] Array;
    delete[] Idx;
    return false;
  }

  if (rank == 0)
  {
    world.cout0("Initializing YGM data members");
  }

  world.barrier();

  // initiate the random array
  srand(time(NULL) + rank);

  // Initialize the target array
  if( size == 1 ){
    for( unsigned i=0; i<iters; i++ ){
      Target[i] = 0;
    }
  }else if( this->GetBenchType() == CT_PTRCHASE ){
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      Target[i] = (int)(rand()%(size));
    }
  }else{
    for( unsigned i=0; i<iters; i++ ){
      // make ring of target PE's 
      if( rank == (size-1) ){
        // last pe
        Target[i] = 0;
      }else{
        Target[i] = rank + 1;
      }
    }
  }

  // setup the Idx values
  for( unsigned i=0; i<(iters+1); i++ ){
    if( this->GetBenchType() == CT_PTRCHASE ){
      Idx[i] = (uint64_t)(rand()%(iters-1));
    }else{
      Idx[i] = (uint64_t)(rand()%(elems));
    }
  }

  for( uint64_t i=0; i<elems; i++ ){
    Array[i] = (uint64_t)(rand());
  }

  world.barrier();

  if( rank == 0 ){
    std::cout << "Done initializing YGM data members" << std::endl;
  }

  return true;
}

bool CT_YGM::FreeData(){
  
  if( Array ){
    delete[] Array;
  }
  if( Idx ){
    delete[] Idx;
  }
  if( Target ){
    delete[] Target;
  }
  
  yp_Array = nullptr;
  yp_Idx = nullptr;

  return true;
}

#endif

// EOF
