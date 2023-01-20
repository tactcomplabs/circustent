/*
 * _CT_CUDA_CU_
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
    d_Array(nullptr),
    d_Idx(nullptr),
    memSize(0),
    iters(0),
    elems(0),
    stride(0),
    threadBlocks(0),
    threadsPerBlock(0)
    {}

CT_CUDA::~CT_CUDA() {}

bool CT_CUDA::PrintCUDADeviceProperties() {
    
    int device, deviceCount;
    cudaDeviceProp properties;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "            CUDA Device Properties" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    // Get and print the number of CUDA devices on the platform
    if(cudaGetDeviceCount(&deviceCount) != cudaSuccess){
        std::cout << "CT_CUDA::PrintCUDADeviceProperties: cudaGetDeviceCount failed!" << std::endl;
        return false;
    }
    else{
        std::cout << "Number of CUDA enabled devices detected: " << deviceCount << std::endl;   
    }
    
    // Get the target device
    if(cudaGetDevice(&device) != cudaSuccess){
        std::cout << "CT_CUDA::PrintCUDADeviceProperties: cudaGetDevice failed!" << std::endl;
        return false; 
    }
    
    // Get the target device properties
    if(cudaGetDeviceProperties(&properties, device) != cudaSuccess){
        std::cout << "CT_CUDA::PrintCUDADeviceProperties: cudaGetDeviceProperties failed!" << std::endl;
        return false; 
    }
    
    // Print out the target device details
    std::cout << "Target Device Details:" << std::endl;
    std::cout << "Device Name: " << properties.name << std::endl;
    std::cout << "Global Memory (bytes): " << properties.totalGlobalMem << std::endl;
    std::cout << "Compute Capability: " << properties.major << "." << properties.minor << std::endl;
    
    return true;
}

bool CT_CUDA::AllocateData(uint64_t m, uint64_t b, uint64_t t, uint64_t i, uint64_t s) {
    // save the data
    memSize = m;
    threadBlocks = b;
    threadsPerBlock = t;
    iters = i;
    stride = s;

    // check args
    if ( threadBlocks <= 0 ) {
        std::cout << "CT_CUDA::AllocateData: threadBlocks must be greater than 0" << std::endl;
        return false;
    }
    if ( threadsPerBlock <= 0 ) {
        std::cout << "CT_CUDA::AllocateData: threadsPerBlock must be greater than 0" << std::endl;
        return false;
    }
    if ( iters == 0 ) {
        std::cout << "CT_CUDA::AllocateData: `iters` cannot be 0" << std::endl;
        return false;
    }
    if ( stride == 0 ) {
        std::cout << "CT_CUDA::AllocateData: `stride` cannot be 0" << std::endl;
        return false;
    }

    // calculate the number of elements
    elems = (memSize/8);
    uint64_t idxMemSize = (sizeof(uint64_t) * (threadBlocks + 1) * iters);
    uint64_t idxElems = (idxMemSize/8);

    // test to see whether we'll stride out of bounds
    uint64_t end = (threadBlocks * iters * stride) - stride;
    if ( end >= elems ) {
        std::cout << "CT_CUDA::AllocateData : `Array` is not large enough for threadBlocks="
                  << threadBlocks << "; iters=" << iters << "; stride=" << stride << std::endl;
        return false;
    }

    // Allocate arrays on the host  
    Array = (uint64_t *) malloc(memSize);
    if ( Array == nullptr ) {
        std::cout << "CT_CUDA::AllocateData : 'Array' could not be allocated" << std::endl;
        return false;
    }

    Idx = (uint64_t *) malloc(idxMemSize);
    if ( Idx == nullptr ) {
        std::cout << "CT_CUDA::AllocateData : 'Idx' could not be allocated" << std::endl;
        if(Array){
            free(Array);   
        }
        return false;
    }

    // allocate data on the target device
    if ( cudaMalloc(&d_Array, memSize) != cudaSuccess ) {
        std::cout << "CT_CUDA::AllocateData : 'd_Array' could not be allocated on device" << std::endl;
        cudaFree(d_Array);
        if(Array){
            free(Array);   
        }
        if(Idx){
            free(Idx);   
        }
        return false;
    }

    if ( cudaMalloc(&d_Idx, idxMemSize) != cudaSuccess ) {
        std::cout << "CT_CUDA::AllocateData : 'd_Idx' could not be alloced on device" << std::endl;
        cudaFree(d_Array);
        cudaFree(d_Idx);
        if(Array){
            free(Array);   
        }
        if(Idx){
            free(Idx);   
        }
        return false;
    }

    // Randomize the arrays on the host
    srand(time(NULL));
    if ( this->GetBenchType() == CT_PTRCHASE ) {
        for ( unsigned i = 0; i < idxElems; i++ ) {
            Idx[i] = (uint64_t)(rand()%(idxElems - 1));
        }
    }
    else {
        for ( unsigned i = 0; i < idxElems; i++ ) {
            Idx[i] = (uint64_t)(rand()%(elems - 1));
        }
    }
    for ( unsigned i=0; i<elems; i++ ) {
        Array[i] = (uint64_t)(rand());
    }

    // copy arrays from host to target device
    if ( cudaMemcpy(d_Array, Array, memSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
        std::cout << "CT_CUDA::AllocateData : 'd_Array' could not be copied to device" << std::endl;
        cudaFree(d_Array);
        cudaFree(d_Idx);
        if(Array){
            free(Array);   
        }
        if(Idx){
            free(Idx);   
        }
        return false;
    }

    if ( cudaMemcpy(d_Idx, Idx, idxMemSize, cudaMemcpyHostToDevice) != cudaSuccess ) {
        std::cout << "CT_CUDA::AllocateData : 'd_Idx' could not be copied to device" << std::endl;
        cudaFree(d_Array);
        cudaFree(d_Idx);
        if(Array){
            free(Array);   
        }
        if(Idx){
            free(Idx);   
        }
        return false;
    }

    return true;
}

bool CT_CUDA::Execute(double &Timing, double &GAMS) {

    CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
    CTBaseImpl::CTAtomType  AType   = this->GetAtomType();  // atomic type
    double StartTime = 0.; // start time
    double EndTime   = 0.; // end time
    double OPS       = 0.; // billions of operations

    // determine benchmark type and launch the desired kernel
    if ( BType == CT_RAND ) {
        switch ( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                RAND_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            case CT_CAS:
                StartTime = this->MySecond();
                RAND_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_STRIDE1 ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                STRIDE1_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                STRIDE1_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_STRIDEN ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                STRIDEN_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, iters, stride );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                STRIDEN_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, iters, stride );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_PTRCHASE ) {
        /* PTRCHASE kernels use only a single thread per block and,      *
         * as such, threadsPerBlock as specified by the user is ignored. */
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                PTRCHASE_ADD<<< threadBlocks, 1 >>>( d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                PTRCHASE_CAS<<< threadBlocks, 1 >>>( d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_SG ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                SG_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(4, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                SG_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(4, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_CENTRAL ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                CENTRAL_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                CENTRAL_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(1, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_SCATTER ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                SCATTER_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(3, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                SCATTER_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(3, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else if ( BType == CT_GATHER ) {
        switch( AType ) {
            case CT_ADD:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                GATHER_ADD<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(3, iters, threadBlocks);
                break;
            case CT_CAS:
                cudaDeviceSynchronize();
                StartTime = this->MySecond();
                GATHER_CAS<<< threadBlocks, threadsPerBlock >>>( d_Array, d_Idx, iters );
                cudaDeviceSynchronize();
                EndTime   = this->MySecond();
                OPS = this->GAM(3, iters, threadBlocks);
                break;
            default:
                this->ReportBenchError();
                return false;
                break;
        }
    }
    else {
        this->ReportBenchError();
        return false;
    }
    
    Timing = this->Runtime(StartTime,EndTime);
    GAMS   = OPS/Timing;

    return true;
}

bool CT_CUDA::FreeData() {
    if ( Array ) {
        free(Array);
    }
    if ( Idx ) {
        free(Idx);
    }
    if ( d_Array ) {
        cudaFree(d_Array);
    }
    if ( d_Idx ) {
        cudaFree(d_Idx);
    }
    return true;
}

#endif // _CT_CUDA_CUH_

// EOF