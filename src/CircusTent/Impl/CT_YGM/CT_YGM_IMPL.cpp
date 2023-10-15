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

    // for i ← 0 to iters by 1 do
    //     AMO(VAL[IDX[i]])
    // end

    uint64_t start = 0x1;

    for ( uint64_t i = 0; i < iters; i++ ){
        
        // AMO to be performed on VAL at index IDX[i]
        auto add_rand = [](uint64_t index, uint64_t value)
        {
            val[index] += value; 
        };

        // Call AMO on rank that is responsible for VAL[IDX[i]] with ADD AMO value of 0x1.
        // Because all of idx \in [0, pes*elems], idx[i]/elems will tell us the rank that 
        // we need to call the procedure on, and idx[i] % elems will tell us where to perform
        // the operation in that rank's array.
        // This behaves like the distributed array described in the paper.
        world.async(idx[i]/elems, add_rand, idx[i] % elems, start);
    }
}

void CT_YGM::RAND_CAS(){

    // for i ← 0 to iters by 1 do
    //     AMO(VAL[IDX[i]])
    // end

    for ( uint64_t i = 0; i < iters; i++ ){
        
        // AMO to be performed on VAL at index IDX[i]
        auto cas_rand = [](uint64_t index)
        {
            // This looks a bit strange but is similar to the CPP STD implementation
            uint64_t expected = val[index];

            uint64_t desired = val[index] + 1; // This prevents the entire lambda from being optimized to nothing

            if (val[index] == expected)
            {
                val[index] = desired;
            }
        };

        // Call AMO on rank that is responsible for VAL[IDX[i]] with CAS defined above.
        // routing of procedure is the same as in RAND_ADD
        world.async(idx[i]/elems, cas_rand, idx[i] % elems);
    }
}

void CT_YGM::STRIDE1_ADD(){

    // for i ← 0 to iters by 1 do
    //     AMO(VAL[i])
    // end

    uint64_t start = 0xF;

#ifdef _SHORTCUT_RPC_

    // This implementation is more reminiscent of the CPP STD implementation
    // and other shared memory systems

    // Makes much better use of YGM's flexible RPC strengths;
    // One message triggers edits on span of iter. 
    auto add_stride1 = [](uint64_t iter_count, uint64_t value) 
    {
        // Almost like a batch of AMOs, we perform them all at the same time
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            val[i] += value;
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, add_stride1, iters, start);

#else

    /*
    This format may be a closer comparison to other STRIDE1 implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

    // Akin to the MPI implementation, each edit is function call to the 
    // next rank in ring of pes
    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[i]
        auto add_stride1 = [](uint64_t index, uint64_t value) 
        {
            val[index] += value;
        };

        // Send AMO to next in pe ring to be run
        world.async(target, add_stride1, i, start);
    }    

#endif
}

void CT_YGM::STRIDE1_CAS(){
    
    // for i ← 0 to iters by 1 do
    //     AMO(VAL[i])
    // end

    uint64_t start = 0xF;

    // @SEE CT_YGM::STRIDE1_ADD() for comparison of different benchmark implementations

#ifdef _SHORTCUT_RPC_

    auto cas_stride1 = [](uint64_t iter_count, uint64_t expected, uint64_t desired) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            if (val[i] == expected)
            {
                val[i] = desired;
            }
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, cas_stride1, iters, start, start);

#else

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[i]
        auto cas_stride1 = [](uint64_t index, uint64_t expected, uint64_t desired)
        {
            // This type of CAS, where expected and desired are not preloaded, is more
            // similar to the MPI implementation
            if (val[index] == expected)
            {
                val[index] = desired;
            }
        };

        // Send AMO to next in pe ring
        world.async(target, cas_stride1, i, start, start);
    }

#endif
}

void CT_YGM::STRIDEN_ADD(){
    
    // for i ← 0 to iters by stride do
    //         AMO(VAL[i])
    // end

    uint64_t start = 0xF;

#ifdef _SHORTCUT_RPC_

    // This implementation is more reminiscent of the CPP STD implementation
    // and other shared memory systems

    // Makes much better use of YGM's flexible RPC strengths
    // one message triggers all edits on span of stride
    auto add_striden = [](uint64_t iter_count, uint64_t stride_len, uint64_t value) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            val[i * stride_len] += value;
        }
    };

    world.async(target, add_striden, iters, stride, start);

#else

    /*
    This format may be a closer comparison to other STRIDEN implementations.
    however, it does not use the full functionality of the 
    YGM RPC paradigm.
    */

   uint64_t ind = 0;

    // Akin to the MPI implementation, each edit is function call to the 
    // targeted rank
    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[ind]
        auto add_striden = [](uint64_t index, uint64_t value)
        { 
            val[index] += value;
        };

        // Send AMO to next in pe ring to be run
        world.async(target, add_striden, ind, start);

        ind += stride;
    }

#endif
}

void CT_YGM::STRIDEN_CAS(){

    // for i ← 0 to iters by stride do
    //         AMO(VAL[i])
    // end

    // @SEE CT_YGM::STRIDEN_ADD() for comparison of different benchmark implementations
    
    uint64_t start = 0xF;

#ifdef _SHORTCUT_RPC_

    auto cas_striden = [](uint64_t iter_count, uint64_t stride_len, uint64_t expected, uint64_t desired) 
    {
        for (uint64_t i = 0; i<iter_count; i++) 
        {
            // AMO to be performed on VAL[i]
            if (val[i*stride_len] == expected)
            {
                val[i*stride_len] = desired;
            }
        }
    };

    // Send batch of AMOs to target rank
    world.async(target, cas_striden, iters, stride, start, start);

#else

    uint64_t ind = 0;

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[idx]
        auto cas_striden = [](uint64_t index, uint64_t expected, uint64_t desired)
        {
            if (val[index] == expected)
            {
                val[index] = desired;
            }
        };

        // Send AMO to next in pe ring
        world.async(target, cas_striden, ind, start, start);

        ind += stride;
    }
    
#endif
}

void CT_YGM::CENTRAL_ADD(){

    // for i ← 0 to iters by 1 do 
    //    AMO(VAL[0])
    // end

    uint64_t start = 0x1;

    // The two different implementations seen here are 
    // similar to the implementations of stride1 and strideN. 
    // Please see those descriptions to get an understanding of 
    // which implementation is preferred for the application.

#ifdef _SHORTCUT_RPC_

    auto add_central = [](uint64_t iter_count, uint64_t value)
    {
        for (uint64_t i = 0; i<iter_count; i++) {
            // AMO to be performed on VAL[0]
            val[0] += value;
        }
    };

    // Send batched AMOs to rank 0, which contains the beginning of the distributed array
    world.async(0, add_central, iters, start);

#else

    for (uint64_t i = 0; i<iters; i++) {

        // AMO to be performed on VAL[0]
        auto add_central = [](uint64_t value)
        {
            val[0] += value;
        };

        // Send AMO to rank 0, which contains the beginning of the distributed array
        world.async(0, add_central, start);
    }

#endif
}

void CT_YGM::CENTRAL_CAS(){

    // for i ← 0 to iters by 1 do 
    //    AMO(VAL[0])
    // end

#ifdef _SHORTCUT_RPC_

    auto cas_central = [](uint64_t iter_count)
    {
        for (uint64_t i = 0; i<iter_count; i++) {

            // This method of presetting expected and desired
            // is similar to CPP STD implementation
            // MPI implementation uses result_buff from previous op and 1.
            uint64_t expected = val[0];

            uint64_t desired = val[0];

            // AMO to be performed on VAL[0]
            if (val[0] == expected)
            {
                val[0] = desired;
            }

        }
    };

    // Send batched AMOs to rank 0, which contains the beginning of the distributed array
    world.async(0, cas_central, iters);

#else

    for (uint64_t i = 0; i < iters; i++) {

        auto cas_central = []()
        {
            // This method of presetting expected and desired
            // is similar to CPP STD implementation
            // MPI implementation uses result_buff from previous op and 1.
            uint64_t expected = val[0];

            uint64_t desired = val[0] + 1;

            // AMO to be performed on VAL[0]
            if (val[0] == expected)
            {
                val[0] = desired;
            }
        };

        // Send AMO to rank 0, which contains the beginning of the distributed array
        world.async(0, cas_central);
    }

#endif

}

void CT_YGM::PTRCHASE_ADD(){

    uint64_t zero = 0;

    // This benchmark uses recursive procedure call to traverse the array
    // AMO and recursive behavior is defined in @SEE: CT_YGM::chase_functor_add

    // start the ptrchase from the beginning of the local idx values. It will traverse elsewhere as it 
    // chases throughout the distributed array
    for( uint64_t i = 0; i < chasers_per_rank; i++ ){
        world.async(idx[i]/(iters + 1), chase_functor_add(), idx[i] % (iters + 1), zero, iters);
    }

}

void CT_YGM::PTRCHASE_CAS(){

    uint64_t zero = 0;

    // This benchmark uses recursive procedure call to traverse the array
    // AMO and recursive behavior is defined in @SEE: CT_YGM::chase_functor_cas

    // start the ptrchase from the beginning of the local idx values. It will traverse elsewhere as it 
    // chases throughout the distributed array
    for( uint64_t i = 0; i < chasers_per_rank; i++ ){
        world.async(idx[i]/(iters + 1), chase_functor_cas(), idx[i] % (iters + 1), zero, iters);
    }

}

void CT_YGM::SCATTER_ADD(){

    // for i ← 0 to iters by 1 do
    //     dest = AMO(IDX[i+1])
    //     val = AMO(VAL[i])
    //     AMO(VAL[dest], val) // VAL[dest] = val
    // end

    uint64_t dest = 0;
    uint64_t start = 0;

    auto add_scatter = [](uint64_t index, uint64_t value)
    {
        val[index] += value;
    };

    for( uint64_t i = 0; i < iters; i++ )
    {
        // dest = AMO(IDX[i+1])
        dest = idx[i + 1] + (uint64_t)(0x0);

        // val = AMO(VAL[i])
        start = val[i] + (uint64_t)(0x0);

        world.async(dest / elems, add_scatter, dest % elems, start);
    }
}

void CT_YGM::SCATTER_CAS(){

    // for i ← 0 to iters by 1 do
    //     dest = AMO(IDX[i+1])
    //     val = AMO(VAL[i])
    //     AMO(VAL[dest], val) // VAL[dest] = val
    // end

    uint64_t dest = 0;
    uint64_t start  = 1;

    // this seems much closer to the original intent of the kernel but
    // less of a comparison to the mpi kernel 

    auto cas_scatter = [](uint64_t index, uint64_t value)
    {
        // perform CAS at destination with val
        if (val[index] == value)
        {
            val[index] = 0;
        }
    };

    for( uint64_t i = 0; i < iters; i++ ){

        // These CAS are somewhat similar to MPI implementation
        // CAS for dest = AMO(IDX[i+1])
        if( idx[i + 1] == 0 ){
            idx[i + 1] = 0;
        }
        dest = idx[i + 1];

        // CAS for val = AMO(VAL[i])
        if( val[i] == 1 ){
            val[i] = 1;
        }
        start = val[i];

        world.async(dest / elems, cas_scatter, dest % elems, start);
    }  
}

void CT_YGM::GATHER_ADD(){

    // uses a middleman RPC (gather_add_request) to return VAL[src] back to calling rank
    /*    execution sequence:
     *      Sender (rank N)  ->       Modifier (rank M)      ->      Sender  (rank N)
     *      async(request)   |    request() - async(reply)   |          reply()
     */

    uint64_t src = 0;

    static auto gather_add_reply = [](uint64_t index, uint64_t value)
    {
        // performs AMO(VAL[i], val) at origin rank
        val[index] += value;
    };

    auto gather_add_request = [](auto pcomm, uint64_t index, uint64_t iter, uint64_t sender)
    {
        // val_src = AMO(VAL[src])
        uint64_t val_src = val[index] + (uint64_t)(0x1);

        pcomm->async(sender, gather_add_reply, iter, val_src);
    };

    for(uint64_t i = 0; i < iters; i++){
        // src = AMO(IDX[i+1])
        src = idx[i + 1] + (uint64_t)(0x0);

        // send reply to perform AMO at requester VAL[i] with remote value VAL[src]
        world.async(src / elems, gather_add_request, src % elems, i, rank);
    }

}

void CT_YGM::GATHER_CAS(){

    // uses a middleman RPC (gather_cas_request) to return VAL[src] back to calling rank
    /*    execution sequence:
     *      Sender (rank N)  ->       Modifier (rank M)      ->      Sender  (rank N)
     *      async(request)   |    request() - async(reply)   |          reply()
     */

    uint64_t src = 0;

    static auto gather_cas_reply = [](uint64_t index, uint64_t value)
    {
        // performs CAS AMO(VAL[i], val) at origin rank
        if( val[index] == value ){
            // use of zero for desired is like MPI
            val[index] = 0;
        }
    };

    auto gather_cas_request = [](auto pcomm, uint64_t index, uint64_t iter, uint64_t sender)
    {
        uint64_t val_src = 0x0;

        // CAS for val
        if( val[index] == 0 ){
            val[index] = 0;
        }
        val_src = val[index];

        pcomm->async(sender, gather_cas_reply, iter, val_src);
    };

    for(uint64_t i = 0; i < iters; i++){
        // src = AMO(IDX[i+1])
        if (idx[i] == 0)
        {
            idx[i] = 0;
        }
        src = idx[i + 1];

        // send request to perform AMO at local VAL[i] with remote value VAL[src]
        world.async(src / elems, gather_cas_request, src % elems, i, rank);
    }
}

void CT_YGM::SG_ADD(){

    // uses a middleman RPC (sg_add_request) to fwd VAL[src] and AMO to destination rank
    /*    execution sequence:
     *      Sender (rank N)  ->       Modifier (rank M)      ->       Reciever  (rank K)
     *      async(request)   |    request() - async(reply)   |          reply()
     */

    uint64_t src = 0;
    uint64_t dest = 0;

    static auto sg_add_fwd = [](uint64_t index, uint64_t value)
    {
        // performs AMO(VAL[i], val) at origin rank
        val[index] += value;
    };

    auto sg_add_request = [](auto pcomm, uint64_t val_index, uint64_t reciever, uint64_t amo_index)
    {
        // val = AMO(VAL[src])
        uint64_t val_src = val[val_index] + (uint64_t)(0x0);
      
        // now that we know val,
        // send a call for AMO(VAL[dest], val) to owner of VAL[dest]
        pcomm->async(reciever, sg_add_fwd, amo_index, val_src);
    };

    for(uint64_t i = 0; i < iters; i++){

        // dest = AMO(IDX[i+1])
        dest = idx[i + 1] + (uint64_t)(0x0);

        // src = AMO(IDX[i])
        src = idx[i] + (uint64_t)(0x0);

        // The other two AMO's are remote and must be routed according to src and dest
        // local -> src (pickup val from src) -> perform amo at dest
        world.async(src / elems, sg_add_request, src % elems, dest / elems, dest % elems);
    }
}

void CT_YGM::SG_CAS(){

    // uses a middleman RPC (sg_cas_request) to fwd VAL[src] and AMO to destination rank
    /*    execution sequence:
     *      Sender (rank N)  ->       Modifier (rank M)      ->       Reciever  (rank K)
     *      async(request)   |    request() - async(reply)   |          reply()
     */

    uint64_t src = 0;
    uint64_t dest = 0;

    static auto sg_cas_fwd = [](uint64_t index, uint64_t value)
    {
        // CAS at VAL[dest] with VAL[src]
        if( val[index] == value ){
            // again, swap with zero is similar to MPI implementation
            val[index] = 0;
        }
    };

    auto sg_cas_request = [](auto pcomm, uint64_t val_index, uint64_t reciever, uint64_t amo_index)
    {
        uint64_t val_src = 0x0;

        // val = AMO(VAL[src])
        if( val[val_index] == 0 ){
            val[val_index] = 0;
        }
        val_src = val[val_index];
      
        // now that we know val,
        // send a call for AMO(VAL[dest], val) to owner of VAL[dest]
        pcomm->async(reciever, sg_cas_fwd, amo_index, val_src);
    };

    for(uint64_t i = 0; i < iters; i++){

        // CAS for dest = AMO(IDX[i+1])
        if (idx[i + 1] == 0)
        {
            idx[i + 1] = 0;
        }
        dest = idx[i + 1];

        // CAS for src = AMO(IDX[i])
        if (idx[i] == 0)
        {
            idx[i] = 0;
        }
        src = idx[i];

        // The other two AMO's are remote and must be routed according to src and dest
        // local -> src (pickup val from src) -> perform amo at dest
        world.async(src / elems, sg_cas_request, src % elems, dest / elems, dest % elems);
    }
}

// EOF
