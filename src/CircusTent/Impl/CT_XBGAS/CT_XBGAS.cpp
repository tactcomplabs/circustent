//
// _CT_XBGAS_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_XBGAS.h"

#ifdef _CT_XBGAS_H_

CT_XBGAS::CT_XBGAS(CTBaseImpl::CTBenchType B,
               CTBaseImpl::CTAtomType A) : CTBaseImpl("XBGAS",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           Target(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0) {
}

CT_XBGAS::~CT_XBGAS(){
}

bool CT_XBGAS::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type
  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations

  if( xbrtime_mype() == 0 ){
    std::cout << "Beginning test execution" << std::endl;
  }

  // determine the benchmark type
  if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      xbrtime_barrier();
      StartTime = this->MySecond();
      RAND_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      RAND_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      STRIDE1_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      STRIDE1_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      STRIDEN_ADD( Array, Idx, Target, iters, pes, stride );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      STRIDEN_CAS( Array, Idx, Target, iters, pes, stride );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      PTRCHASE_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      PTRCHASE_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      SG_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      SG_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      CENTRAL_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      CENTRAL_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      SCATTER_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      SCATTER_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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
      xbrtime_barrier();
      StartTime = this->MySecond();
      GATHER_ADD( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      xbrtime_barrier();
      StartTime = this->MySecond();
      GATHER_CAS( Array, Idx, Target, iters, pes );
      xbrtime_barrier();
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

bool CT_XBGAS::AllocateData( uint64_t m,
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
    std::cout << "CT_XBGAS::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_XBGAS::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_XBGAS::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (iters * stride)-stride;
  if( end >= elems ){
    std::cout << "CT_XBGAS::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << "; stride =" << stride
              << std::endl;
    return false;
  }

  // ensure that we have enough allocation space
  if( !(this->GetBenchType() == CT_PTRCHASE) ){
    if( elems < iters ){
      std::cout << "CT_XBGAS::AllocateData : Memory size is too small for iteration count" << std::endl;
      std::cout << "                       : Increase the memory footprint per PE or reduce the iteration count" << std::endl;
      return false;
    }
  }

  // init the xbgas context
  xbrtime_init();

  if( xbrtime_mype() == 0 ){
    std::cout << "xBGAS Info:  Vendor = Tactical Computing Laboratories"
              << "; PEs = " << xbrtime_num_pes() << std::endl;
  }

  xbrtime_barrier();

  // 'Array' resides in symmetric heap space
  Array = (uint64_t *)(xbrtime_malloc( memSize ));
  if( Array == nullptr ){
    std::cout << "CT_XBGAS::AllocateData : 'Array' could not be allocated" << std::endl;
    xbrtime_close();
    return false;
  }

  // 'Idx' resides in symmetric heap space
  Idx = (uint64_t *)(xbrtime_malloc( sizeof(uint64_t) * (iters+1) ));
  if( Idx == nullptr ){
    std::cout << "CT_XBGAS::AllocateData : 'Idx' could not be allocated" << std::endl;
    xbrtime_free( Array );
    xbrtime_close();
    return false;
  }

  // 'Target' resides in local PE memory
  Target = (int *)(malloc( sizeof( int ) * iters ));
  if( Target == nullptr ){
    std::cout << "CT_XBGAS:AllocateData: 'Target' could not be allocated" << std::endl;
    xbrtime_free(Array);
    xbrtime_free(Idx);
    xbrtime_close();
    return false;
  }

  if( xbrtime_mype() == 0 ){
    std::cout << "Initializing XBGAS data members" << std::endl;
  }

  xbrtime_barrier();

  // initiate the random array
  srand(time(NULL) + xbrtime_mype());

  // Init the target array
  if( xbrtime_num_pes() == 1 ){
    for( unsigned i=0; i<iters; i++ ){
      Target[i] = 0;
    }
  }else if( (this->GetBenchType() == CT_PTRCHASE) || 
            (this->GetBenchType() == CT_RAND) || 
            (this->GetBenchType() == CT_SCATTER) || 
            (this->GetBenchType() == CT_GATHER) ||
            (this->GetBenchType() == CT_SG) ){
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      Target[i] = (int)(rand()%(xbrtime_num_pes()-1));
    }
  }else{
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      if( xbrtime_mype() == (xbrtime_mype()-1) ){
        // last pe
        Target[i] = 0;
      }else{
        Target[i] = xbrtime_mype() + 1;
      }
    }
  }

  // setup the Idx and array values
  for( unsigned i=0; i<(iters+1); i++ ){
    Idx[i] = (uint64_t)(rand()%(elems));
  }
  for( uint64_t i=0; i<elems; i++ ){
    Array[i] = (uint64_t)(rand());
  }

  if( xbrtime_mype() == 0 ){
    std::cout << "Done initializing XBGAS data members" << std::endl;
  }

  xbrtime_barrier();

  return true;
}

bool CT_XBGAS::FreeData(){
  if( Array ){
    xbrtime_free( Array );
  }
  if( Idx ){
    xbrtime_free( Idx );
  }

  free( Target );

  return true;
}

#endif

// EOF
