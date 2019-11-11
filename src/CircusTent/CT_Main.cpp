//
// _CT_Main_cpp_
//
// Copyright (C) 2017-2019 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CircusTent/CircusTent.h"
#ifdef _ENABLE_OMP_
#include "Impl/CT_OMP/CT_OMP.h"
#endif
#ifdef _ENABLE_OPENSHMEM_
#include <mpp/shmem.h>
#include "Impl/CT_OPENSHMEM/CT_SHMEM.h"
#endif

void PrintTiming( double Timing, double GAMS );

#ifdef _ENABLE_OPENSHMEM_
void RunBenchOpenSHMEM( CTOpts *Opts ){
  // init the OpenMP object
  CT_SHMEM *CT = new CT_SHMEM(Opts->GetBenchType(),
                              Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_SHMEM OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_SHMEM" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_SHMEM" << std::endl;
    CT->FreeData();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_SHMEM" << std::endl;
    free( CT );
    return ;
  }

  // Print the timing
  if( shmem_my_pe() == 0 ){
    PrintTiming( Timing, GAMS );
  }
  shmem_finalize();
}
#endif


#ifdef _ENABLE_OMP_
void RunBenchOMP( CTOpts *Opts ){
  // init the OpenMP object
  CT_OMP *CT = new CT_OMP(Opts->GetBenchType(),
                          Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_OMP OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_OMP" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OMP" << std::endl;
    CT->FreeData();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OMP" << std::endl;
    free( CT );
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );
}
#endif

void PrintTiming(double Timing, double GAMS){
  std::cout << "================================================" << std::endl;
  std::cout << " Timing (secs)        : " << Timing << std::endl;
  std::cout << " Giga AMOs/sec (GAMS) : " << GAMS << std::endl;
  std::cout << "================================================" << std::endl;
}

int main( int argc, char **argv ){
  CTOpts *Opts = new CTOpts();

  if( !Opts->ParseOpts(argc,argv) ){
    std::cout << "Failed to parse command line options" << std::endl;
    delete Opts;
    return -1;
  }

  if( (!Opts->IsHelp()) && (!Opts->IsList()) ){
    // execute the benchmarks
#ifdef _ENABLE_OMP_
    RunBenchOMP(Opts);
#endif
#ifdef _ENABLE_OPENSHMEM_
    RunBenchOpenSHMEM(Opts);
#endif
  }

  delete Opts;
  return 0;
}

// EOF
