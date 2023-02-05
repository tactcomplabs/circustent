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
                                           //Array(nullptr),
                                           //Idx(nullptr),
                                           //Target(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0) {
}

CT_YGM::~CT_YGM(){
}

bool CT_YGM::Execute(double &Timing, double &GAMS){
  return true;
}

bool CT_YGM::AllocateData( uint64_t m,
                           uint64_t p,
                           uint64_t i,
                           uint64_t s){
  return true;
}

bool CT_YGM::FreeData(){
  return true;
}

#endif

// EOF
