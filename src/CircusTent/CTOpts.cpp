//
// _CTOpts_cpp_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CircusTent/CTOpts.h"

BenchType BenchTypeTable[] = {
  { "RAND_ADD",    "", "Random memory access pattern using FETCH+ADD",        CTBaseImpl::CT_RAND,    CTBaseImpl::CT_ADD, false, false },
  { "RAND_CAS",    "", "Random memory access pattern using CAS"      ,        CTBaseImpl::CT_RAND,    CTBaseImpl::CT_CAS, false, false },
  { "STRIDE1_ADD", "", "Stride-1 memory access pattern usign FETCH+ADD",      CTBaseImpl::CT_STRIDE1, CTBaseImpl::CT_ADD, false, false },
  { "STRIDE1_CAS", "", "Stride-1 memory access pattern usign CAS",            CTBaseImpl::CT_STRIDE1, CTBaseImpl::CT_CAS, false, false },
  { "STRIDEN_ADD", "stride", "Stride-N memory access pattern usign FETCH+ADD",CTBaseImpl::CT_STRIDEN, CTBaseImpl::CT_ADD, false, true },
  { "STRIDEN_CAS", "stride", "Stride-N memory access pattern usign CAS",      CTBaseImpl::CT_STRIDEN, CTBaseImpl::CT_CAS, false, true },
  { "PTRCHASE_ADD","", "Pointer chase memory access pattern using FETCH+ADD", CTBaseImpl::CT_PTRCHASE,CTBaseImpl::CT_ADD, false, false },
  { "PTRCHASE_CAS","", "Pointer chase memory access pattern using CAS",       CTBaseImpl::CT_PTRCHASE,CTBaseImpl::CT_CAS, false, false },
  { "CENTRAL_ADD", "", "Centralized point access using FETCH+ADD",            CTBaseImpl::CT_CENTRAL, CTBaseImpl::CT_ADD, false, false },
  { "CENTRAL_CAS", "", "Centralized point access using CAS",                  CTBaseImpl::CT_CENTRAL, CTBaseImpl::CT_CAS, false, false },
  { "SG_ADD",      "", "Scatter/Gather memory access pattern using FETCH+ADD",CTBaseImpl::CT_SG,      CTBaseImpl::CT_ADD, false, false },
  { "SG_CAS",      "", "Scatter/Gather memory access pattern using CAS",      CTBaseImpl::CT_SG,      CTBaseImpl::CT_CAS, false, false },
  { "SCATTER_ADD", "", "Scatter memory access pattern using FETCH+ADD",       CTBaseImpl::CT_SCATTER, CTBaseImpl::CT_ADD, false, false },
  { "SCATTER_CAS", "", "Scatter memory access pattern using CAS",             CTBaseImpl::CT_SCATTER, CTBaseImpl::CT_CAS, false, false },
  { "GATHER_ADD", "",  "Gather memory access pattern using FETCH+ADD",        CTBaseImpl::CT_GATHER,  CTBaseImpl::CT_ADD, false, false },
  { "GATHER_CAS", "",  "Gather memory access pattern using CAS",              CTBaseImpl::CT_GATHER,  CTBaseImpl::CT_CAS, false, false },
  { ".",           "", "."                                           ,        CTBaseImpl::CT_NB,      CTBaseImpl::CT_NA,  false, false } // disable flag
};

CTOpts::CTOpts()
  : isHelp(false), isList(false), memSize(0), iters(0), stride(1),
    l_argc(0), l_argv(nullptr) {
}

CTOpts::~CTOpts(){
}

CTBaseImpl::CTBenchType CTOpts::GetBenchType(){
  unsigned Idx = 0;
  while( BenchTypeTable[Idx].Name != "." ){
    if( BenchTypeTable[Idx].Enabled ){
      BenchTypeTable[Idx].Enabled = true;
      return BenchTypeTable[Idx].BType;
    }
    Idx++;
  }
  return CTBaseImpl::CT_NB;
}

CTBaseImpl::CTAtomType CTOpts::GetAtomType(){
  unsigned Idx = 0;
  while( BenchTypeTable[Idx].Name != "." ){
    if( BenchTypeTable[Idx].Enabled ){
      BenchTypeTable[Idx].Enabled = true;
      return BenchTypeTable[Idx].AType;
    }
    Idx++;
  }
  return CTBaseImpl::CT_NA;
}

bool CTOpts::EnableBench( std::string Bench ){
  unsigned Idx = 0;
  while( BenchTypeTable[Idx].Name != "." ){
    if( Bench == BenchTypeTable[Idx].Name ){
      BenchTypeTable[Idx].Enabled = true;
      return true;
    }
    Idx++;
  }
  std::cout << "Unknown benchmark type: " << Bench << std::endl;
  return false;
}

bool CTOpts::ParseOpts(int argc, char **argv){
  l_argc = argc;
  l_argv = argv;
  for( int i=1; i<argc; i++ ){
    std::string s(argv[i]);

    if( (s=="-h") || (s=="-help") || (s=="--help") ){
      isHelp = true;
      PrintHelp();
      return true;
    }
    else if( (s=="-a") || (s=="-arch") || (s=="--arch") ){
      isHelp = true;
      PrintArch();
      return true;
    }
    else if( (s=="-b") || (s=="-bench") || (s=="--bench") ){
      if( i+1 > (argc-1) ){
        std::cout << "Error : --bench requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      if( !EnableBench( P ) )
        return false;
      i++;
    }
    else if( (s=="-m") || (s=="-memsize") || (s=="--memsize") ){
      if( i+1 > (argc-1) ){
        std::cout << "Error : --memsize requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      memSize = atoll(P.c_str());
      i++;
    }
    else if( (s=="-i") || (s=="-iters") || (s=="--iters") ){
      if( i+1 > (argc-1) ){
        std::cout << "Error : --iters requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      iters = atoll(P.c_str());
      i++;
    }
    else if( (s=="-s") || (s=="-stride") || (s=="--stride") ){
      if( i+1 > (argc-1) ){
        std::cout << "Error : --stride requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      stride = atoll(P.c_str());
      if( stride == 0 ){
        std::cout << "Error : --stride cannot be 0" << std::endl;
        return false;
      }
      i++;
    }
    else if( (s=="-l") || (s=="-list") || (s=="--list") ){
      isList = true;
      PrintBench();
      return true;
    }
#if defined(_ENABLE_CUDA_)
    else if( (s=="-o") || (s=="-blocks") || (s=="--blocks")){
        if ( i+1 > (argc-1)) {
            std::cout << "Error: --blocks requires an argument" << std::endl;
            return false;
        }
        std::string P(argv[i+1]);
        threadBlocks = atoi(P.c_str());
        i++;
    }
    else if ((s=="-t") || (s=="-threads") || (s=="--threads")){
        if ( i+1 > (argc-1)) {
            std::cout << "Error: --threads requires an argument" << std::endl;
            return false;
        }
        std::string P(argv[i+1]);
        threadsPerBlock = atoi(P.c_str());
        i++;
    }
#else
    else if( (s=="-p") || (s=="-pes") || (s=="--pes") ){
      if( i+1 > (argc-1) ){
        std::cout << "Error : --pes requires an argument" << std::endl;
        return false;
      }
      std::string P(argv[i+1]);
      pes = atoll(P.c_str());
      i++;
    }
#endif
    else{
      std::cout << "Unknown option: " << s << std::endl;
      return false;
    }
  }

  // sanity check the options
  if( memSize == 0 ){
    std::cout << "Error : memory size cannot be 0" << std::endl;
    return false;
  }

  return true;
}

void CTOpts::PrintArch(){
  std::cout << "===================================================================================" << std::endl;
  std::cout << " CircusTent Configuration" << std::endl;
  std::cout << "===================================================================================" << std::endl;

  // target configuration
#ifdef _ENABLE_OMP_
  std::cout << " Arch = OpenMP" << std::endl;
#endif
#ifdef _ENABLE_OMP_TARGET_
  std::cout << " Arch = OpenMP Target" << std::endl;
#endif
#ifdef _ENABLE_OPENSHMEM_
  std::cout << " Arch = OpenSHMEM" << std::endl;
#endif
#ifdef _ENABLE_MPI_
  std::cout << " Arch = MPI" << std::endl;
#endif
#ifdef _ENABLE_XBGAS_
  std::cout << " Arch = xBGAS" << std::endl;
#endif
#ifdef _ENABLE_PTHREADS_
  std::cout << " Arch = Pthreads" << std::endl;
#endif
#ifdef _ENABLE_OPENACC_
  std::cout << " Arch = OpenACC" << std::endl;
#endif
#ifdef _ENABLE_OPENCL_
  std::cout << " Arch = OpenCL" << std::endl;
#endif
#ifdef _ENABLE_CPP_STD_
  std::cout << " Arch = CXX Threads" << std::endl;
#endif
#ifdef _ENABLE_CUDA_
  std::cout << " Arch = CUDA" << std::endl;
#endif

  // compiler
#ifdef __clang__
  std::cout << " Compiler = Clang " << __clang_major__ << "." << __clang_minor__ << std::endl;
#elif defined(__GNUC__)
  std::cout << " Compiler = GCC " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#endif

// platform
#ifdef __x86_64__
  std::cout << " Platform = x86_64" << std::endl;
#elif __arm__
  std::cout << " Platform = arm" << std::endl;
#elif __powerpc
  std::cout << " Platform = PowerPC" << std::endl;
#elif __zarch__
  std::cout << " Platform = SystemZ" << std::endl;
#else
  std::cout << " Platform = Unknown" << std::endl;
#endif

  // C++ version
  long cpp_version = __cplusplus;
  if (cpp_version == 199711L) {
    std::cout << " C++ Version = C++98/03" << std::endl;
  } else if (cpp_version == 201103L) {
    std::cout << " C++ Version = C++11" << std::endl;
  } else if (cpp_version == 201402L) {
    std::cout << " C++ Version = C++14" << std::endl;
  } else if (cpp_version == 201703L) {
    std::cout << " C++ Version = C++17" << std::endl;
  } else if (cpp_version > 201703L) {
    std::cout << " C++ Version = C++20 or later" << std::endl;
  }
  std::cout << "===================================================================================" << std::endl;
}

void CTOpts::PrintBench(){
  std::cout << "===================================================================================" << std::endl;
  std::cout << "BENCHMARK | REQUIRED_OPTIONS | DESCRIPTION" << std::endl;
  std::cout << "===================================================================================" << std::endl;
  unsigned Idx = 0;
  while( BenchTypeTable[Idx].Name != "." ){
    std::cout << " - " << BenchTypeTable[Idx].Name;
    if( BenchTypeTable[Idx].ReqArg ){
      std::cout << " | " << BenchTypeTable[Idx].Arg << " | ";
    }else{
      std::cout << " | No Arg Required | ";
    }
    std::cout << BenchTypeTable[Idx].Notes << std::endl;
    Idx++;
  }
  std::cout << "===================================================================================" << std::endl;
}

void CTOpts::PrintHelp(){
  unsigned major = CT_VERSION_MAJOR;
  unsigned minor = CT_VERSION_MINOR;
  std::cout << "=================================================================================" << std::endl;
  std::cout << " CircusTent Version " << major << "." << minor << std::endl;
  std::cout << " Usage: circustent [OPTIONS]" << std::endl;
  std::cout << "=================================================================================" << std::endl;
  std::cout << " -b|-bench|--bench TEST                    : Sets the benchmark to run" << std::endl;
  std::cout << " -m|-memsize|--memsize BYTES               : Sets the size of the array" << std::endl;
  std::cout << " -i|-iters|--iters ITERATIONS              : Sets the number of iterations per PE" << std::endl;
  std::cout << " -s|-stride|--stride STRIDE (elems)        : Sets the stride in 'elems'" << std::endl;
#if defined(_ENABLE_CUDA_)
  std::cout << " -o|-blocks|--blocks THREAD_BLOCKS         : Sets the number of thread blocks" << std::endl;
  std::cout << " -t|-threads|--threads THREADS_PER_BLOCK   : Sets the number of threads per block" << std::endl;
#else
  std::cout << " -p|-pes|--pes PES                         : Sets the number of PEs" << std::endl;
#endif
  std::cout << "=================================================================================" << std::endl;
  std::cout << " -h|-help|--help                           : Prints this help menu" << std::endl;
  std::cout << " -l|-list|--list                           : List benchmarks" << std::endl;
  std::cout << " -a|-arch|--arch                           : Prints the benchmark configuration" << std::endl;
  std::cout << "=================================================================================" << std::endl;
}

// EOF
