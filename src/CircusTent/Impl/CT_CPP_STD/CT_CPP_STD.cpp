//
// _CT_CPP_STD_CPP_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CT_CPP_STD.h"

#ifdef _CT_CPP_STD_H_

CT_CPP_STD::CT_CPP_STD(CTBaseImpl::CTBenchType B,
                       CTBaseImpl::CTAtomType A) :
                         CTBaseImpl("CPP_STD",B,A),
                         Array(nullptr),
                         Idx(nullptr),
                         memSize(0),
                         pes(0),
                         iters(0),
                         elems(0),
                         stride(0) {
}

CT_CPP_STD::~CT_CPP_STD(){
}

bool CT_CPP_STD::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type
  uint64_t i        = 0;  // loop var
  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations
  std::atomic<std::uint64_t> barrier_ctr(0); // Counter for kernel start barrier
  std::thread threads[pes];

  // determine the benchmark type
  if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::RAND_ADD, this, i, &barrier_ctr, &StartTime);

      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::RAND_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::STRIDE1_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::STRIDE1_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::STRIDEN_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::STRIDEN_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::PTRCHASE_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::PTRCHASE_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::SG_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(4,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::SG_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::CENTRAL_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::CENTRAL_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::SCATTER_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::SCATTER_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::GATHER_ADD, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
      EndTime   = this->MySecond();
      OPS = this->GAM(3,iters,pes);
      break;
    case CT_CAS:
      for(i = 0; i < pes; i++){
        threads[i] = std::thread(&CT_CPP_STD::GATHER_CAS, this, i, &barrier_ctr, &StartTime);
      }
      JoinThreads(threads);
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

  barrier_ctr = 0;
  Timing = this->Runtime(StartTime,EndTime);
  GAMS   = OPS/Timing;

  return true;
}

bool CT_CPP_STD::AllocateData(uint64_t m,
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
    std::cout << "CT_CPP_STD::AllocateData : 'pes' cannot be 0" << std::endl;
    return false;
  }
  if( iters == 0 ){
    std::cout << "CT_CPP_STD::AllocateData : 'iters' cannot be 0" << std::endl;
    return false;
  }
  if( stride == 0 ){
    std::cout << "CT_CPP_STD::AllocateData : 'stride' cannot be 0" << std::endl;
    return false;
  }

  // calculate the number of elements
  elems = (memSize/8);

  // test to see whether we'll stride out of bounds
  uint64_t end = (pes * iters * stride) - stride;
  if( end >= elems ){
    std::cout << "CT_CPP_STD::AllocateData : 'Array' is not large enough for pes="
              << pes << "; iters=" << iters << ";stride =" << stride
              << std::endl;
    return false;
  }

  Array =  new (std::nothrow) std::atomic<std::uint64_t>[elems];
  if( Array == nullptr ){
    std::cout << "CT_CPP_STD::AllocateData : 'Array' could not be allocated" << std::endl;
    return false;
  }

  Idx =  new (std::nothrow) std::atomic<std::uint64_t>[(pes+1)*iters];
  if( Idx == nullptr ){
    std::cout << "CT_CPP_STD::AllocateData : 'Idx' could not be allocated" << std::endl;
    delete[] Array;
    return false;
  }

  expected = new (std::nothrow) uint64_t[iters*pes];
  if( expected == nullptr ) {
    std::cout << "CT_CPP_STD::AllocateData : 'expected' could not be allocated" << std::endl;
    delete[] expected;
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

  std::cout << "RUNNING WITH NUM_THREADS = " << pes << std::endl;

  return true;
}

bool CT_CPP_STD::FreeData(){
  if( Array ){
    delete[] Array;
  }
  if( Idx ){
    delete[] Idx;
  }
  if( expected ){
    delete[] expected;
  }

  return true;
}

void CT_CPP_STD::JoinThreads(std::thread *threads){
  uint64_t i;
  for(i = 0; i < pes; i++){
    threads[i].join();
  }
}

void CT_CPP_STD::MyBarrier(std::atomic<std::uint64_t> *barrier_ctr){

  // Increment barrier_ctr
  barrier_ctr->fetch_add(1, std::memory_order_relaxed);

  // Spin until barrier_ctr == pes
  while(barrier_ctr->load(std::memory_order_relaxed) != pes){};
}

#endif

// EOF
