//
// _CT_SHMEM_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_SHMEM.h"

#ifdef _CT_SHMEM_H_

CT_SHMEM::CT_SHMEM(CTBaseImpl::CTBenchType B,
               CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENSHMEM",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           Target(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0) {
}

CT_SHMEM::~CT_SHMEM(){
}

bool CT_SHMEM::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type
  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations

  if( shmem_my_pe() == 0 ){
    std::cout << "Beginning test execution" << std::endl;
  }

  // determine the benchmark type
  if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      shmem_barrier_all();
      StartTime = this->MySecond();
      RAND_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      RAND_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      STRIDE1_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      STRIDE1_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      STRIDEN_ADD( Array, Idx, Target, iters, pes, stride );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      STRIDEN_CAS( Array, Idx, Target, iters, pes, stride );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      PTRCHASE_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      PTRCHASE_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      SG_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      SG_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      CENTRAL_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      CENTRAL_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      SCATTER_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      SCATTER_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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
      shmem_barrier_all();
      StartTime = this->MySecond();
      GATHER_ADD( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      shmem_barrier_all();
      StartTime = this->MySecond();
      GATHER_CAS( Array, Idx, Target, iters, pes );
      shmem_barrier_all();
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

bool CT_SHMEM::AllocateData( uint64_t m,
                           uint64_t p,
                           uint64_t i,
                           uint64_t s){
  // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // allocate all the memory
  if( pes == 0 ){
    std::cout << "CT_SHMEM::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_SHMEM::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_SHMEM::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (iters * stride)-stride;
  if( end > elems ){
    std::cout << "CT_SHMEM::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  // ensure that we have enough allocation space
  if( !(this->GetBenchType() == CT_PTRCHASE) ){
    if( elems < iters ){
      std::cout << "CT_SHMEM::AllocateData : Memory size is too small for iteration count" << std::endl;
      std::cout << "                       : Increase the memory footprint per PE or reduce the iteration count" << std::endl;
      return false;
    }
  }

  // init the shmem context
  shmem_init();

  // print the shmem version info
  int major = 0;
  int minor = 0;
  char vendor[SHMEM_MAX_NAME_LEN];
  shmem_info_get_version(&major,&minor);
  shmem_info_get_name(&vendor[0]);

  if( shmem_my_pe() == 0 ){
    std::cout << "OpenSHMEM Info:  Vendor = " << vendor
              << "; Version = " << major << "." << minor
              << "; PEs = " << shmem_n_pes() << std::endl;
  }

  shmem_barrier_all();

  // 'Array' resides in symmetric heap space
  Array = (uint64_t *)(shmem_malloc( memSize ));
  if( Array == nullptr ){
    std::cout << "CT_SHMEM::AllocateData : 'Array' could not be allocated" << std::endl;
    shmem_finalize();
    return false;
  }

  // 'Idx' resides in symmetric heap space
  Idx = (uint64_t *)(shmem_malloc( sizeof(uint64_t) * (iters+1) ));
  if( Idx == nullptr ){
    std::cout << "CT_SHMEM::AllocateData : 'Idx' could not be allocated" << std::endl;
    shmem_free( Array );
    shmem_finalize();
    return false;
  }

  // 'Target' resides in local PE memory
  Target = (int *)(malloc( sizeof( int ) * iters ));
  if( Target == nullptr ){
    std::cout << "CT_SHMEM:AllocateData: 'Target' could not be allocated" << std::endl;
    shmem_free(Array);
    shmem_free(Idx);
    shmem_finalize();
    return false;
  }

  if( shmem_my_pe() == 0 ){
    std::cout << "Initializing SHMEM data members" << std::endl;
  }

  shmem_barrier_all();

  // initiate the random array
  srand(time(NULL) + shmem_my_pe());

  // Init the target array
  if( shmem_n_pes() == 1 ){
    for( unsigned i=0; i<iters; i++ ){
      Target[i] = 0;
    }
  }else if( this->GetBenchType() == CT_PTRCHASE ){
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      Target[i] = (int)(rand()%(shmem_n_pes()));
    }
  }else{
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      if( shmem_my_pe() == (shmem_n_pes()-1) ){
        // last pe
        Target[i] = 0;
      }else{
        Target[i] = shmem_my_pe() + 1;
      }
    }
  }

  // setup the Idx and array values
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

  if( shmem_my_pe() == 0 ){
    std::cout << "Done initializing SHMEM data members" << std::endl;
  }

  shmem_barrier_all();

  return true;
}

bool CT_SHMEM::FreeData(){
  if( Array ){
    shmem_free( Array );
  }
  if( Idx ){
    shmem_free( Idx );
  }

  free( Target );

  return true;
}

#endif

// EOF
