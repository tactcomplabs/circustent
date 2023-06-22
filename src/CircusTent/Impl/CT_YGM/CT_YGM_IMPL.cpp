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

void CT_YGM::RAND_ADD(){

    uint64_t start = 0x1;

    for ( uint64_t i = 0; i < iters; i++ ){
        
        // AMO to be performed on VAL at index IDX[i]
        auto add_rand = [](auto parray, uint64_t index, uint64_t value)
        {
            (*parray)[index] += value; 
        };

        // Call AMO on rank that is responsible for VAL[IDX[i]] with ADD AMO value of 0x1.
        // Because all of Idx \in [0, pes*elems], Idx[i]/elems will tell us the rank that 
        // we need to call the procedure on, and Idx[i] % elems will tell us where to perform
        // the operation in that rank's array.
        // This behaves like the distributed array described in the paper.
        world.async(Idx[i]/elems, add_rand, yp_Array, Idx[i] % elems, start);
    }
}

void CT_YGM::RAND_CAS(){
    
    uint64_t start = 0x1;

    for ( uint64_t i = 0; i < iters; i++ ){
        
        // AMO to be performed on VAL at index IDX[i]
        auto cas_rand = [](auto parray, uint64_t index)
        {
            // This looks a bit strange but is similar to the CPP STD implementation
            uint64_t expected = (*parray)[index];

            uint64_t desired = (*parray)[index];

            if ((*parray)[index] == expected)
            {
                (*parray)[index] = desired;
            }
        };

        // Call AMO on rank that is responsible for VAL[IDX[i]] with CAS defined above.
        // routing of procedure is the same as in RAND_ADD
        world.async(Idx[i]/elems, cas_rand, yp_Array, Idx[i] % elems);
    }
}

void CT_YGM::STRIDE1_ADD(){

    uint64_t start = 0xF;

#ifdef _NAIVE_RPC_YGM_

    /*
    This format may be a closer comparison to other STRIDE1 implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

    // Akin to the MPI implementation, each edit is function call to the 
    // next rank in ring of pes
    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[i]
        auto add_stride1 = [](auto parray, uint64_t index, uint64_t value) 
        {
            (*parray)[index] += value;
        };

        // Send AMO to next in pe ring to be run
        world.async(target, add_stride1, yp_Array, i, start);
    }

#else

    // This implementation is more reminiscent of the CPP STD implementation
    // and other shared memory systems

    // Makes much better use of YGM's flexible RPC strengths;
    // One message triggers edits on span of iter. 
    auto add_stride1 = [](auto parray, uint64_t iter_count, uint64_t value) 
    {
        // Almost like a batch of AMOs, we perform them all at the same time
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            (*parray)[i] += value;
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, add_stride1, yp_Array, iters, start);

#endif
}

void CT_YGM::STRIDE1_CAS(){
    
    uint64_t start = 0xF;

    // @SEE CT_YGM::STRIDE1_ADD() for comparison of different benchmark implementations

#ifdef _NAIVE_RPC_YGM_

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[i]
        auto cas_stride1 = [](auto parray, uint64_t index, uint64_t expected, uint64_t desired)
        {
            // This type of CAS, where expected and desired are not preloaded, is more
            // similar to the MPI implementation
            if ((*parray)[index] == expected)
            {
                (*parray)[index] = desired;
            }
        };

        // Send AMO to next in pe ring
        world.async(target, cas_stride1, yp_Array, i, start, start);
    }

#else

    auto cas_stride1 = [](auto parray, uint64_t iter_count, uint64_t expected, uint64_t desired) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            if ((*parray)[i] == expected)
            {
                (*parray)[i] = desired;
            }
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, cas_stride1, yp_Array, iters, start, start);

#endif
}

void CT_YGM::STRIDEN_ADD(){
    
    uint64_t start = 0xF;

#ifdef _NAIVE_RPC_YGM_

    /*
    This format may be a closer comparison to other STRIDEN implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

   uint64_t idx = 0;

    // Akin to the MPI implementation, each edit is function call to the 
    // targeted rank
    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[idx]
        auto add_striden = [](auto parray, uint64_t index, uint64_t value)
        { 
            (*parray)[index] += value;
        };

        // Send AMO to next in pe ring to be run
        world.async(target, add_striden, yp_Array, idx, start);

        idx += stride;
    }

#else

    // This implementation is more reminiscent of the CPP STD implementation
    // and other shared memory systems

    // Makes much better use of YGM's flexible RPC strengths
    // one message triggers all edits on span of stride
    auto add_striden = [](auto parray, uint64_t iter_count, uint64_t stride_len, uint64_t value) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            (*parray)[i * stride_len] += value;
        }
    };

    world.async(target, add_striden, yp_Array, iters, stride, start);

#endif
}

void CT_YGM::STRIDEN_CAS(){

    // @SEE CT_YGM::STRIDEN_ADD() for comparison of different benchmark implementations
    
    uint64_t start = 0xF;

#ifdef _NAIVE_RPC_YGM_

    uint64_t idx = 0;

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[idx]
        auto cas_striden = [](auto parray, uint64_t index, uint64_t expected, uint64_t desired)
        {
            if ((*parray)[index] == expected)
            {
                (*parray)[index] = desired;
            }
        };

        // Send AMO to next in pe ring
        world.async(target, cas_striden, yp_Array, idx, start, start);

        idx += stride;
    }

#else

    auto cas_striden = [](auto parray, uint64_t iter_count, uint64_t stride_len, uint64_t expected, uint64_t desired) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            if ((*parray)[i*stride_len] == expected)
            {
                (*parray)[i*stride_len] = desired;
            }
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, cas_striden, yp_Array, iters, stride, start, start);
    
#endif
}

void CT_YGM::CENTRAL_ADD(){

    uint64_t start = 0x1;

    // The two different implementations seen here are 
    // similar to the implementations of stride1 and strideN. 
    // Please see those descriptions to get an understanding of 
    // which implementation is preferred for the application.

#ifdef _NAIVE_RPC_YGM_

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[0]
        auto add_central = [](auto parray, uint64_t value)
        {
            (*parray)[0] += value;
        };

        // Send AMO to rank 0, which contains the beginning of the distributed array
        world.async(0, add_central, yp_Array, start);
    }

#else

    auto add_central = [](auto parray, uint64_t iter_count, uint64_t value)
    {
        for (uint64_t i = 0; i<iter_count; i++) {
            // AMO to be performed on VAL[0]
            (*parray)[0] += value;
        }
    };

    // Send batched AMOs to rank 0, which contains the beginning of the distributed array
    world.async(0, add_central, yp_Array, iters, start);

#endif
}

void CT_YGM::CENTRAL_CAS(){

#ifdef _NAIVE_RPC_YGM_

    for (uint64_t i = 0; i < iters; i++) {

        auto cas_central = [](auto parray)
        {
            // This method of presetting expected and desired
            // is similar to CPP STD implementation
            // MPI implementation uses result_buff from previous op and 1.
            uint64_t expected = (*parray)[0];

            uint64_t desired = (*parray)[0];

            // AMO to be performed on VAL[0]
            if ((*parray)[0] == expected)
            {
                (*parray)[0] = desired;
            }
        };

        // Send AMO to rank 0, which contains the beginning of the distributed array
        world.async(0, cas_central, yp_Array);
    }

#else

    auto cas_central = [](auto parray, uint64_t iter_count)
    {
        for (uint64_t i = 0; i<iter_count; i++) {

            // This method of presetting expected and desired
            // is similar to CPP STD implementation
            // MPI implementation uses result_buff from previous op and 1.
            uint64_t expected = (*parray)[0];

            uint64_t desired = (*parray)[0];

            // AMO to be performed on VAL[0]
            if ((*parray)[0] == expected)
            {
                (*parray)[0] = desired;
            }

        }
    };

    // Send batched AMOs to rank 0, which contains the beginning of the distributed array
    world.async(0, cas_central, yp_Array, iters);

#endif

}

void CT_YGM::PTRCHASE_ADD(){

    uint64_t zero = 0;

    // This benchmark uses recursive procedure call to traverse the array
    // AMO and recursive behavior is defined in @SEE: CT_YGM::chase_functor_add

    // start the ptrchase from the beginning of the local Idx values. It will traverse elsewhere as it 
    // chases throughout the distributed array
    world.async(Idx[0]/(iters + 1), chase_functor_add(), yp_Idx, Idx[0] % (iters + 1), zero, iters, iters + 1);

}

void CT_YGM::PTRCHASE_CAS(){

    uint64_t zero = 0;

    // This benchmark uses recursive procedure call to traverse the array
    // AMO and recursive behavior is defined in @SEE: CT_YGM::chase_functor_cas

    // start the ptrchase from the beginning of the local Idx values. It will traverse elsewhere as it 
    // chases throughout the distributed array
    world.async(Idx[0]/(iters + 1), chase_functor_cas(), yp_Idx, Idx[0] % (iters + 1), zero, iters, iters + 1);

}

void CT_YGM::SCATTER_ADD(){

    // First implementation of Scatter Kernel, uses a remote Idx.
    // Similar to the MPI that always targets a remote rank with the AMO operations, but
    // following that route appears less true to what is described in the paper when
    // compared to CT::YGM_SCATTER_ADD_ALTERNATE when considering the 
    // intent of the kernel

    for( uint64_t i = 0; i < iters; i++ )
    {
        world.async(target, CT_YGM::scatter_functor_add(), yp_Array, yp_Idx, i, elems);
    }
}

void CT_YGM::SCATTER_ADD_ALTERNATE(){

    // this seems much closer to the original intent of the kernel but
    // less of a comparison to the mpi kernel 

    uint64_t dest = 0;
    uint64_t val = 0;

    for( uint64_t i = 0; i < iters; i++ )
    {
        // dest = AMO(IDX[i+1])
        dest = Idx[i + 1] + (uint64_t)(0x0);

        // val = AMO(VAL[i])
        val = Array[i] + (uint64_t)(0x0);

        auto add_scatter = [](auto parray, uint64_t index, uint64_t value)
        {
          (*parray)[index] += value;
        };

        world.async(dest / elems, add_scatter, yp_Array, dest % elems, val);
    }
}

void CT_YGM::SCATTER_CAS(){

    // First implementation of Scatter Kernel, uses a remote Idx.
    // Similar to the MPI that always targets a remote rank with the AMO operations, but
    // following that route appears less true to what is described in the paper when
    // compared to CT::YGM_SCATTER_CAS_ALTERNATE when considering the 
    // intent of the kernel
    for( uint64_t i = 0; i < iters; i++ ){
        world.async(target, CT_YGM::scatter_functor_cas(), yp_Array, yp_Idx, i, elems);
    }  
}

void CT_YGM::SCATTER_CAS_ALTERNATE(){

    uint64_t dest = 0;
    uint64_t val  = 1;

    // this seems much closer to the original intent of the kernel but
    // less of a comparison to the mpi kernel 

    for( uint64_t i = 0; i < iters; i++ ){

        // These CAS are somewhat similar to MPI implementation
        // CAS for dest = AMO(IDX[i+1])
        if( Idx[i + 1] == 0 ){
            Idx[i + 1] = 0;
        }
        dest = Idx[i + 1];

        // CAS for val = AMO(VAL[i])
        if( Array[i] == 1 ){
            Array[i] = 1;
        }
        val = Array[i];

        auto cas_scatter = [](auto parray, uint64_t index, uint64_t value)
        {
          // perform CAS at destination with val
          if ((*parray)[index] == value)
          {
            (*parray)[index] = 0;
          }
        };

        world.async(dest / elems, cas_scatter, yp_Array, dest % elems, val);
    }  
}

void CT_YGM::GATHER_ADD(){

    // uses a middleman RPC to return VAL[src] back to calling rank
    // @SEE: CT_YGM::gather_functor_add

    // for large # of iterations, could become non-deterministic
    // due to message timing on network

    uint64_t src = 0;

    for(uint64_t i = 0; i < iters; i++){
        // src = AMO(IDX[i+1])
        src = Idx[i + 1] + (uint64_t)(0x0);

        // send request to perform AMO at local VAL[i] with remote value VAL[src]
        world.async(src / elems, gather_functor_add(), yp_Array, src % elems, i, rank);
    }

}

void CT_YGM::GATHER_CAS(){

    // uses a middleman RPC to return VAL[src] back to calling rank
    // @SEE: CT_YGM::gather_functor_cas

    uint64_t src = 0;

    for(uint64_t i = 0; i < iters; i++){
        // src = AMO(IDX[i+1])
        if (Idx[i] == 0)
        {
            Idx[i] = 0;
        }
        src = Idx[i+1];

        // send request to perform AMO at local VAL[i] with remote value VAL[src]
        world.async(src / elems, gather_functor_cas(), yp_Array, src % elems, i, rank);
    }
}

void CT_YGM::SG_ADD(){

    uint64_t src = 0;
    uint64_t dest = 0;

    for(uint64_t i = 0; i < iters; i++){

        // dest = AMO(IDX[i+1])
        dest = Idx[i + 1] + (uint64_t)(0x0);

        // src = AMO(IDX[i])
        src = Idx[i] + (uint64_t)(0x0);

        // The other two AMO's are remote and must be routed according to src and dest
        // local -> src (pickup val from src) -> perform amo at dest
        world.async(src / elems, sg_functor_add(), yp_Array, src % elems, dest / elems, dest % elems);
    }
}

void CT_YGM::SG_CAS(){

    uint64_t src = 0;
    uint64_t dest = 0;

    for(uint64_t i = 0; i < iters; i++){

        // CAS for dest = AMO(IDX[i+1])
        if (Idx[i + 1] == 0)
        {
            Idx[i + 1] = 0;
        }
        dest = Idx[i + 1];

        // CAS for src = AMO(IDX[i])
        if (Idx[i] == 0)
        {
            Idx[i] = 0;
        }
        src = Idx[i];

        // The other two AMO's are remote and must be routed according to src and dest
        // local -> src (pickup val from src) -> perform amo at dest
        world.async(src / elems, sg_functor_cas(), yp_Array, src % elems, dest / elems, dest % elems);
    }
}

// EOF
