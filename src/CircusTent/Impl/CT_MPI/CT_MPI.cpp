//
// _CT_MPI_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
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
  uint64_t end = (iters * stride)-stride;
  if( end >= elems ){
    std::cout << "CT_MPI::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << "; stride =" << stride
              << std::endl;
    return false;
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
  MPI_Barrier(MPI_COMM_WORLD);

  // 'Array' resides in local heap space
  MPI_Alloc_mem(memSize, MPI_INFO_NULL, &Array);

  // Create Array window
  MPI_Win_create(Array, memSize, sizeof(uint64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &ArrayWin);

  // 'Idx' resides in local heap space
  MPI_Alloc_mem(sizeof(uint64_t) * (iters+1), MPI_INFO_NULL, &Idx);

  // Create Idx window
  MPI_Win_create(Idx, sizeof(uint64_t) * (iters+1), sizeof(uint64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &IdxWin);

  // 'Target' resides in local PE memory
  Target = (int *)(malloc( sizeof( int ) * iters ));
  if( Target == nullptr ){
    std::cout << "CT_MPI:AllocateData: 'Target' could not be allocated" << std::endl;
    MPI_Win_free(&ArrayWin);
    MPI_Win_free(&IdxWin);
    MPI_Free_mem(Array);
    MPI_Free_mem(Idx);
    MPI_Finalize();
    return false;
  }

  if( rank == 0 ){
    std::cout << "Initializing MPI data members" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // mersenne_twister_engine seeded uniquely for each rank
  std::mt19937_64 gen(pes + rank);

  // uniform distribution for target array
  std::uniform_int_distribution<int> target_dist(0, size - 1);

  if( rank == 0 ){
    std::cout << "Target distribution MAX (int): " << (int)target_dist.max() << std::endl;
    std::cout << "Target distribution MIN (int): " << (int)target_dist.min() << std::endl;
  }

  // Init the target array
  if( size == 1 ){
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
      Target[i] = (int)target_dist(gen);

      if ((Target[i] > (int)(size - 1)) || (Target[i] < 0))
      {
        std::cout << "Errors int Target[] initialization: values exceeded bounds" << std::endl;
      }
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

  // If this is not the pointer chase benchmark simply point to a random VAL index
  uint64_t max_index = elems - 1;

  // Otherwise, we are pointing to the next value in the distributed IDX array
  if( this->GetBenchType() == CT_PTRCHASE ){
    max_index = iters - 2;
  }

  // uniform distribution for IDX array
  std::uniform_int_distribution<uint64_t> idx_dist(0, max_index);

  if( rank == 0 ){
    std::cout << "IDX distribution MAX (uint64_t): " << (uint64_t) idx_dist.max() << std::endl;
    std::cout << "IDX distribution MIN (uint64_t): " << (uint64_t) idx_dist.min() << std::endl;
  }

  // setup the Idx and array values
  for( uint64_t i=0; i<(iters+1); i++ ){
    Idx[i] = (uint64_t)idx_dist(gen); 
  }

  for( uint64_t i=0; i<elems; i++ ){
    Array[i] = gen();
  }

  if( rank == 0 ){
    std::cout << "Done initializing MPI data members" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool CT_MPI::FreeData(){
  MPI_Barrier(MPI_COMM_WORLD);
  int rank          = -1; // mpi rank
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if( rank == 0 )
    std::cout << "Releasing Memory" << std::endl;

  // Free the window
  MPI_Win_free(&ArrayWin);
  MPI_Win_free(&IdxWin);

  // Free the memory
  MPI_Free_mem(Array);
  MPI_Free_mem(Idx);

  free( Target );

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

#endif

// EOF
