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

// init static parameters. Must be done here so they are accessible 
// by lambdas/functors. Because there is one CT_YGM object per PE for 
// CircusTent, there will not be any conflicts.

uint64_t* CT_YGM::val = nullptr;
uint64_t* CT_YGM::idx = nullptr;
uint64_t CT_YGM::iters = 0;

CT_YGM::CT_YGM(CTBaseImpl::CTBenchType B,
               CTBaseImpl::CTAtomType A) : CTBaseImpl("YGM",B,A),
                                           target(0),
                                           memSize(0),
                                           pes(0),
                                           elems(0),
                                           stride(0),
                                           rank(-1),
                                           chasers_per_rank(1),
                                           world(NULL, NULL) {
}

CT_YGM::~CT_YGM(){}

bool CT_YGM::Execute(double &Timing, double &GAMS){

  CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
  CTBaseImpl::CTAtomType AType    = this->GetAtomType();  // atomic type

  double StartTime  = 0.; // start time
  double EndTime    = 0.; // end time
  double OPS        = 0.; // billions of operations

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
  else if( BType == CT_RAND ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      RAND_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      RAND_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_PTRCHASE ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      PTRCHASE_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      PTRCHASE_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_SCATTER ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      SCATTER_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      SCATTER_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_GATHER ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      GATHER_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      GATHER_CAS();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    default:
      this->ReportBenchError();
      return false;
    }
  }
  else if( BType == CT_SG ){
    switch( AType ){
    case CT_ADD:
      world.barrier();
      StartTime = this->MySecond();
      SG_ADD();
      world.barrier();
      EndTime   = this->MySecond();
      OPS = this->GAM(1,iters,pes);
      break;
    case CT_CAS:
      world.barrier();
      StartTime = this->MySecond();
      SG_CAS();
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

  world.barrier();

  int size          = -1; // ygm size (num pe's)

  // save the benchmark setup data
  memSize = m;
  pes = p;
  iters = i;
  stride = s;

  rank = world.rank();
  size = world.size();

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
    std::cout << "CT_YGM::AllocateData : 'val' is not large enough for pes="
              << pes << "; iters=" << iters << "; stride =" << stride
              << std::endl;
    return false;
  }

  // 'val' resides in local heap space
  val = new std::uint64_t[elems];
  if ( val == nullptr ){
    world.cout("CT_YGM::AllocateData : 'val' could not be allocated");
    return false;
  }

  // 'idx' resides in local heap space
  idx = new std::uint64_t[iters + 1];

  if ( idx == nullptr ){
    world.cout("CT_YGM::AllocateData : 'idx' could not be allocated");
    delete[] val;
    return false;
  }

  if (rank == 0)
  {
    world.cout0("Initializing YGM data members");
  }

  // Initialize the target
  if( size == 1 ){
    target = 0;
  }else{
    // make ring of target PE's 
    target = (rank + 1) % size;
  }

  // Last index on last rank for the distributed array VAL. 
  // informs our distribution for random IDX values
  uint64_t max_val_index = elems;

  if( this->GetBenchType() == CT_PTRCHASE ){
    max_val_index = (pes * (iters + 1)) - 1;
  }else if( (this->GetBenchType() == CT_RAND) ||
            (this->GetBenchType() == CT_SCATTER) ||
            (this->GetBenchType() == CT_GATHER) ||
            (this->GetBenchType() == CT_SG) ){
    max_val_index = (pes * elems) - 1;
  }

  // mersenne_twister_engine seeded uniquely for each rank
  std::mt19937_64 gen(pes + rank);
  std::uniform_int_distribution<uint64_t> ind_dist(0, max_val_index);

  // setup the idx values
  // if benchmark does not depend on idx do not instantiate
  if( !((this->GetBenchType() == CT_STRIDE1) || 
        (this->GetBenchType() == CT_STRIDEN) || 
        (this->GetBenchType() == CT_CENTRAL)))
  {
    for( uint64_t i=0; i<(iters+1); i++ ){
      idx[i] = ind_dist(gen);
    }
  }

  for( uint64_t i=0; i<elems; i++ ){
    val[i] = gen();
  }

  // No need for full ygm::comm barrier, we haven't made any calls yet
  world.cf_barrier(); 

  if( rank == 0 ){
    std::cout << "Done initializing YGM data members" << std::endl;
  }

  return true;
}

bool CT_YGM::FreeData(){
  
  if( val ){
    delete[] val;
  }
  if( idx ){
    delete[] idx;
  }

  return true;
}

void CT_YGM::PrintVal(){
  std::stringstream ss;

  ss << "{";

  for (std::size_t j = 0; j < elems - 1; j++)
  {
      ss << val[j] << ", ";
  }

  ss << val[elems - 1] << "}";

  world.cout(ss.str());
}

void CT_YGM::PrintIdx(){

  std::stringstream ss;

  ss << "{";

  for (std::size_t j = 0; j < iters; j++)
  {
      ss << idx[j] << ", ";
  }

  ss << idx[iters] << "}";

  world.cout(ss.str());
}

#endif

// EOF
