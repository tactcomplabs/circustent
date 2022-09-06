/*
 * FIXME: ensure proper file exension is used throughout this file
 * _CT_CUDA_CU
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_CUDA.cuh"
#ifdef _CT_CUDA_CUH_



CT_CUDA::CT_CUDA(CTBaseImpl::CTBenchType B, CTBaseImpl::CTAtomType A) :
    CTBaseImpl("CUDA", B, A),
    Array(nullptr),
    Idx(nullptr),
    memSize(0),
    pes(0),
    iters(0),
    elems(0),
    stride(0),
    deviceID(-1)
    {}

CT_CUDA::~CT_CUDA() {}

// TODO: helper functions

bool CT_CUDA::Execute(double &Timing, double &GAMS) {
    // TODO: CT_CUDA::Execute()

    CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
    CTBaseImpl::CTAtomType  AType   = this->GetAtomType();  // atomic type

    double StartTime = 0.; // start time
    double Endtime   = 0.; // end time
    double OPS       = 0.; // billions of operations

    // TODO: determine benchmark type
}

bool CT_CUDA::AllocateData(uint64_t m, uint64_t p, uint64_t i, uint64_t s) {
    
}

#endif // _CT_CUDA_CUH_