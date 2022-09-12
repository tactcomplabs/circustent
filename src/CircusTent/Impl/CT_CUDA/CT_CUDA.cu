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




// Kernels meant to be run on the device
__global__  void RAND_ADD(
    uint64_t *ARRAY,
    uint64_t *IDX,
    uint64_t iters,
    uint64_t pes
) {
    // TODO: RAND_ADD()
}

CT_CUDA::CT_CUDA(CTBaseImpl::CTBenchType B, CTBaseImpl::CTAtomType A) :
    CTBaseImpl("CUDA", B, A),
    Array(nullptr),
    Idx(nullptr),
    memSize(0),
    pes(0),
    iters(0),
    elems(0),
    stride(0),
    // TODO: make sure these are all set by the user
    deviceID(-1),
    blocksPerGrid(-1),
    threadsPerBlock(-1)
    {}

CT_CUDA::~CT_CUDA() {}

// helper functions
bool parseCUDAOpts(int argc, char **argv) {
    l_argc = argc;
    l_argv = argv;
    for (int i=1; i < argc; i++) {
        std::string s(argv[i]);

        if ( (s=="-bpg") || (s=="-blocks") || (s=="--blocks") ) {
            if ( i+1 > (argc-1) ) {
                std::cout << "Error: --blocks requires an argument" << std::endl;
                return false;
            }
            std::string P(argv[i+1]);
            blocksPerGrid = atoll(P.c_str()); // FIXME: check this
            i++;
        }
        else if (s=="-tpb") || (s=="-threads") || (s=="--threads") {
            if ( i+1 > (argc-1) ) {
                std::cout << "Error: --threads requires an argument" << std::endl;
                return false;
            }
            std::string P(argv[i+1]);
            threadsPerBlock = atoll(P.c_str()); // FIXME: check this
            i++;
        }
    }

    // sanity check the options
    if ( blocksPerGrid <= 0 ) { // FIXME: check this
        std::cout << "Error: --blocks must be greater than 0" << std::endl;
        return false;
    }

    if ( threadsPerBlock <= 0 ) { // FIXME: check this
        std::cout << "Error: --threads must be greater than 0" << std::endl;
        return false;
    }

    return true;
}

void printDeviceProperties(int deviceID) { // TODO: printDeviceProperties()
    
}



bool CT_CUDA::Execute(double &Timing, double &GAMS) { // TODO: CT_CUDA::Execute()

    CTBaseImpl::CTBenchType BType   = this->GetBenchType(); // benchmark type
    CTBaseImpl::CTAtomType  AType   = this->GetAtomType();  // atomic type

    double StartTime = 0.; // start time
    double Endtime   = 0.; // end time
    double OPS       = 0.; // billions of operations

    // TODO: determine benchmark type and launch the desired kernel
}

bool CT_CUDA::AllocateData(uint64_t m, uint64_t p, uint64_t i, uint64_t s) {
    // save the data
    memSize = m;
    pes = p;
    iters = i;
    srtide = s;

    // check args
    if ( pes == 0 ) {
        std::cout << "CT_CUDA::AllocateData: `pes` cannot be 0" << std::endl; // FIXME: do we need pes for CUDA?
        return false;
    }
    if ( iters == 0 ) {
        std::cout << "CT_CUDA::AllocateData `iters` cannot be 0" << std::endl;
        return false;
    }
    if ( stride == 0 ) {
        std::cout << "CT_CUDA::AllocateData `stride` cannot be 0" << std::endl;
        return false;
    }

    // calculate the number of elements
    elems = (memSize/8);

    // test to see whether we'll stride out of bounds
    uint64_t end = (pes * iters * stride) - stride;
    if ( end > elems ) {
        std::cout << "CT_CUDA::AllocateData : `Array` is not large enough for pes="
        << pes << "; iters=" << iters << "; stride=" << stride << std::endl;
        return false;
    }

    // Allocate arrays on the host  
    Array = (uint64_t *) malloc(memSize);
    if ( Array = nullptr ) {
        std::cout << "CT_CUDA::AllocateData : 'Array' could not be allocated" << std::endl;
        free(Array);
        return false;
    }

    Idx = (uint64_t *) malloc(memSize);
    if ( Idx = nullptr ) {
        std::cout << "CT_CUDA::AllocateData : 'Idx' could not be allocated" << std::endl;
        free(Idx);
        return false;
    }

    // Randomize the arrays on the host
    srand(time(NULL));
    if ( this->GetBenchType() == CT_PTRCHASE ) {
        for ( unsigned i = 0; i < ((pes+1) * iters); i++ ) {
            Idx[i] = (uint64_t)(rand()%((pes+1)*iters));
        }
    }
    else {
        for ( unsigned i = 0; i < ((pes+1) * iters); i++ ) {
            Idx[i] = (uint64_t)(rand()%(elems-1));
        }
    }
    for ( unsigned i=0; i<elems; i++ ) {
        HostArray[i] = (uint64_t)(rand());
    }

    // FIXME: allocate data on the target device
    // TODO: Use cudaSuccess to check that this process is successful, otherwise print an error message
    cudaMalloc(&d_Array, memSize);
    cudaMalloc(&d_Idx, memSize); 


    // FIXME: copy arrays from host to target device
    // TODO: Use cudaSuccess to check that this process is successful, otherwise print an error message
    // cudaMemcpy(d_Array, &Array, memSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_Idx, &Idx, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Array, Array, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Idx, Idx, memSize, cudaMemcpyHostToDevice);

    return true;
}

#endif // _CT_CUDA_CUH_