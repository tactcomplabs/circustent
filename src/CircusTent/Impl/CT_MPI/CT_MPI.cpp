//
// _CT_MPI_CPP_
//
// Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_MPI.h"

#ifdef _CT_MPI_H_

CT_MPI::CT_MPI(CTBaseImpl::CTBenchType B,
               CTBaseImpl::CTAtomType A) : CTBaseImpl("MPI",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           Target(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0) {
}

CT_MPI::~CT_MPI(){
}

bool CT_MPI::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type
  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations
  int rank          = -1; // mpi rank

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if( rank == 0 ){
    std::cout << "Beginning test execution" << std::endl;
  }

  // determine the benchmark type
  if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      RAND_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      RAND_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      STRIDE1_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      STRIDE1_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      STRIDEN_ADD( Array, Idx, Target, iters, pes, stride, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      STRIDEN_CAS( Array, Idx, Target, iters, pes, stride, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      PTRCHASE_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      PTRCHASE_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      SG_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      SG_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      CENTRAL_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      CENTRAL_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      SCATTER_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      SCATTER_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      GATHER_ADD( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      MPI_Barrier(MPI_COMM_WORLD);
      StartTime = this->MySecond();
      GATHER_CAS( Array, Idx, Target, iters, pes, ArrayWin, IdxWin );
      MPI_Barrier(MPI_COMM_WORLD);
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

bool CT_MPI::AllocateData( uint64_t m,
                           uint64_t p,
                           uint64_t i,
                           uint64_t s){

  int rank          = -1; // mpi rank
  int size          = -1; // mpi size (num pe's)

  // save the data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  // allocate all the memory
  if( pes == 0 ){
    std::cout << "CT_MPI::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_MPI::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_MPI::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  if( stride > 1 ){
    uint64_t end = (iters * stride)-stride;
    if( end > elems ){
      std::cout << "CT_MPI::AllocateData : 'Array' is not large enough for pes="
                << pes << "; iters=" << iters << ";stride =" << stride
                << std::endl;
      return false;
    }
  }

  // ensure that we have enough allocation space
  if( !(this->GetBenchType() == CT_PTRCHASE) ){
    if( elems < iters ){
      std::cout << "CT_MPI::AllocateData : Memory size is too small for iteration count" << std::endl;
      std::cout << "                       : Increase the memory footprint per PE or reduce the iteration count" << std::endl;
      return false;
    }
  }

  // init the MPI context
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Win_create_dynamic(MPI_INFO_NULL,MPI_COMM_WORLD,&ArrayWin);
  MPI_Win_create_dynamic(MPI_INFO_NULL,MPI_COMM_WORLD,&IdxWin);
  MPI_Barrier(MPI_COMM_WORLD);

  // 'Array' resides in local heap space
  Array = (uint64_t *)(malloc( memSize ));
  if( Array == nullptr ){
    std::cout << "CT_MPI::AllocateData : 'Array' could not be allocated" << std::endl;
    MPI_Finalize();
    return false;
  }
  MPI_Win_attach(ArrayWin,Array,memSize);

  // 'Idx' resides in local heap space
  Idx = (uint64_t *)(malloc( sizeof(uint64_t) * (iters+1) ));
  if( Idx == nullptr ){
    std::cout << "CT_MPI::AllocateData : 'Idx' could not be allocated" << std::endl;
    MPI_Win_detach(ArrayWin,Array);
    free( Array );
    MPI_Win_free(&ArrayWin);
    MPI_Finalize();
    return false;
  }
  MPI_Win_attach(IdxWin,Idx,sizeof(uint64_t) * (iters+1));

  // 'Target' resides in local PE memory
  Target = (int *)(malloc( sizeof( int ) * iters ));
  if( Target == nullptr ){
    std::cout << "CT_MPI:AllocateData: 'Target' could not be allocated" << std::endl;
    MPI_Win_detach(ArrayWin,Array);
    MPI_Win_detach(IdxWin,Idx);
    free(Array);
    free(Idx);
    MPI_Win_free(&ArrayWin);
    MPI_Win_free(&IdxWin);
    MPI_Finalize();
    return false;
  }

  if( rank == 0 ){
    std::cout << "Initializing MPI data members" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // initiate the random array
  srand(time(NULL) + rank);

  // Init the target array
  if( size == 1 ){
    for( unsigned i=0; i<iters; i++ ){
      Target[i] = 0;
    }
  }else if( this->GetBenchType() == CT_PTRCHASE ){
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      Target[i] = (int)(rand()%(size-1));
    }
  }else{
    for( unsigned i=0; i<iters; i++ ){
      // randomize the Target pe
      if( rank == (size-1) ){
        // last pe
        Target[i] = 0;
      }else{
        Target[i] = rank + 1;
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

  if( rank == 0 ){
    std::cout << "Done initializing MPI data members" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool CT_MPI::FreeData(){
  if( Array ){
    MPI_Win_detach(ArrayWin,Array);
    free( Array );
  }
  if( Idx ){
    MPI_Win_detach(IdxWin,Idx);
    free( Idx );
  }

  free( Target );

  MPI_Win_free(&ArrayWin);
  MPI_Win_free(&IdxWin);

  return true;
}

#endif

// EOF
