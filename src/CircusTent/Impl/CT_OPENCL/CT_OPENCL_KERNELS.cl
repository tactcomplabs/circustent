#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_kr

__kernel void RAND_ADD(
        // XXX: int64_t not available in .cl files
        __global uint *restrict ARRAY,
        __global uint *restrict IDX,
        __global uint* iters,
        __global uint* pes
)
{
    uint i = 0;
    uint start = 0;
    #pragma ocl parallel private(star, i)
    {
        start = (uint)(get_global_size(0)) * iters;
    }

}