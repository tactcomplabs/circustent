// XXX: uint64_t not available in .cl files
// XXX: get_global_id() arguments might need to be adjusted

__kernel void RAND_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
)
{
    ulong i = 0;
    ulong start = 0;
    #pragma ocl parallel private(start, i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for (i=start; i<(start+iters); i++) {
            __atomic_fetch_add( &ARRAY[IDX[i]], (ulong)(0x1), __ATOMIC_RELAXED );
        }
    }
}

__kernel void RAND_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
        __atomic_compare_exchange_n( &ARRAY[IDX[i]], &ARRAY[IDX[i]], ARRAY[IDX[i]],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void STRIDE1_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
            __atomic_fetch_add( &ARRAY[i], (ulong)(0xF), __ATOMIC_RELAXED );
        }
    }
}

__kernel void STRIDE1_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
            __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void STRIDEN_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes,
        ulong stride
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i+=stride ){
        __atomic_fetch_add( &ARRAY[i], (ulong)(0xF), __ATOMIC_RELAXED );
        }
    }
}

__kernel void STRIDEN_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes,
        ulong stride
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i+=stride ){
        __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void PTRCHASE_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=0; i<iters; i++ ){
        start = __atomic_fetch_add( &IDX[start],
                                    (ulong)(0x00ull),
                                    __ATOMIC_RELAXED );
        }
    }
}

__kernel void PTRCHASE_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;

    #pragma ocl parallel private(start,i)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=0; i<iters; i++ ){
        __atomic_compare_exchange_n( &IDX[start], &start, IDX[start],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void SG_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong src    = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,src,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
        src  = __atomic_fetch_add( &IDX[i], (ulong)(0x00ull), __ATOMIC_RELAXED );
        dest = __atomic_fetch_add( &IDX[i+1], (ulong)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[src], (ulong)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
        }
    }
}

__kernel void SG_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong src    = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,src,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        val   = 0x00ull;
        src   = 0x00ull;
        dest  = 0x00ull;
        for( i=start; i<(start+iters); i++ ){
        __atomic_compare_exchange_n( &IDX[i], &src, IDX[i],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[src], &val, ARRAY[src],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void CENTRAL_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;

    #pragma ocl parallel private(i)
    {
        for( i=0; i<iters; i++ ){
        __atomic_fetch_add( &ARRAY[0], (ulong)(0x1), __ATOMIC_RELAXED );
        }
    }
}

__kernel void CENTRAL_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;

    #pragma ocl parallel private(i)
    {
        for( i=0; i<iters; i++ ){
        __atomic_compare_exchange_n( &ARRAY[0], &ARRAY[0], ARRAY[0],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void SCATTER_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
        dest = __atomic_fetch_add( &IDX[i+1], (ulong)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[i], (ulong)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
        }
    }
}

__kernel void SCATTER_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        dest  = 0x00ull;
        val   = 0x00ull;
        for( i=start; i<(start+iters); i++ ){
        __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[i], &val, ARRAY[i],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}

__kernel void GATHER_ADD(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        for( i=start; i<(start+iters); i++ ){
        dest = __atomic_fetch_add( &IDX[i+1], (ulong)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[dest], (ulong)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[i], val, __ATOMIC_RELAXED );
        }
    }
}

__kernel void GATHER_CAS(
        __local ulong* ARRAY,
        __local ulong* IDX,
        ulong iters,
        ulong pes
) {
    // todo
    ulong i      = 0;
    ulong start  = 0;
    ulong dest   = 0;
    ulong val    = 0;

    #pragma ocl parallel private(start,i,dest,val)
    {
        start = (ulong)(get_global_id(0)) * iters;
        dest  = 0x00ull;
        val   = 0x00ull;
        for( i=start; i<(start+iters); i++ ){
        __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[dest], &val, ARRAY[dest],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], val,
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        }
    }
}
