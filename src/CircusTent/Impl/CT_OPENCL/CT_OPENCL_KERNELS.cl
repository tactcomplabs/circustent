#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

__kernel void RAND_ADD(
        // XXX: int64_t not available in .cl files
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
)
{
    // uint i = 0;
    // uint start = 0;
    // #pragma ocl parallel private(star, i)
    // {
    //     start = (uint)(get_global_id(0)) * iters;
    // }
}

__kernel void RAND_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void STRIDE1_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void STRIDE1_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void STRIDEN_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void STRIDEN_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void PTRCHASE_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void PTRCHASE_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void SG_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void SG_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void CENTRAL_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void CENTRAL_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void SCATTER_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void SCATTER_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void GATHER_ADD(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}

__kernel void GATHER_CAS(
        __global uint *ARRAY,
        __global uint *IDX,
        __global uint* iters,
        __global uint* pes
) {
    // todo
}
