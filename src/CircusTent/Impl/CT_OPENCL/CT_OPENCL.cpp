/*
 * TODO: _CT_OPENCL_TARGET_IMPL_C
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_OPENCL.h"

#ifdef _CT_OPENCL_H_

CT_OPENCL::CT_OPENCL(CTBaseImpl::CTBenchType B,
                CTBaseImpl::CTAtomType A) : CTBaseImpl("OPENCL",B,A),
                                           Array(nullptr),
                                           Idx(nullptr),
                                           memSize(0),
                                           pes(0),
                                           iters(0),
                                           elems(0),
                                           stride(0),
                                           deviceTypeStr(""),
                                           deviceID(-1) {
}

CT_OPENCL::~CT_IOENCL() {

}

bool CT_OPENCL::Execute(double &Timing, double &GAMS) {
    // todo
}

bool CT_OPENCL::AllocateData(
    uint64_t m,
    uint64_t p,
    uint64_t i,
    uint64_t s
) {
    // todo
}


bool OPENCL::SetDevice() {
    // todo
}

bool CT_OPENCL::FreeData() {
    // todo
}

#endif

// EOF