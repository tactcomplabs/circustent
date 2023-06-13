/*
 * _CT_YGM_IMPL_CPP_
 *
 * Copyright (C) 2017-2023 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_YGM.h"

void CT_YGM::RAND_ADD( uint64_t iters, uint64_t pes ){
}

void CT_YGM::STRIDE1_ADD(){

    uint64_t start = 0xF;

#ifdef _NAIVE_STRIDE_YGM_

    /*
    This format may be a closer comparison to other STRIDE1 implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

    uint64_t i = 0;
    // Akin to the MPI implementation, each edit is function call to the 
    // targeted rank
    for (i = 0; i<iters; i++) {

        auto add_value = [](auto parray, size_t index, uint64_t value) 
        {
            (*parray)[index] += value;
        };
        
        world.async(Target[i], add_value, yp_Array, i, start);
    }

#else

    // Makes much better use of YGM's flexible RPC strengths
    // one message triggers edits on span of iter. 
    auto add_value = [](auto parray, uint64_t iter_count, uint64_t value) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            (*parray)[i] += value;
        }
    };

    world.async(Target[0], add_value, yp_Array, iters, start);

#endif
}

void CT_YGM::STRIDE1_CAS(){
    return;
}

void CT_YGM::STRIDEN_ADD(){
    
    uint64_t start = 0xF;

#ifdef _NAIVE_STRIDE_YGM_

    /*
    This format may be a closer comparison to other STRIDEN implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

   uint64_t i = 0;
   uint64_t idx = 0;

    // Akin to the MPI implementation, each edit is function call to the 
    // targeted rank
    for (i = 0; i<iters; i++) {

        auto add_value = [](auto parray, size_t index, uint64_t value)
        { 
            (*parray)[index] += value;
        };

        world.async(Target[i], add_value, yp_Array, idx, start);

        idx += stride;
    }

#else

    // Makes much better use of YGM's flexible RPC strengths
    // one message triggers all edits on span of stride
    auto add_value = [](auto parray, uint64_t iter_count, uint64_t stride_len, uint64_t value) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            (*parray)[i * stride_len] += value;
        }
    };

    world.async(Target[0], add_value, yp_Array, iters, stride, start);

#endif
}

void CT_YGM::STRIDEN_CAS(){
    return;
}

// EOF
