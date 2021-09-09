//
// _CT_Main_cpp_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#include "CircusTent/CircusTent.h"

#ifdef _ENABLE_OMP_
#include "Impl/CT_OMP/CT_OMP.h"
#endif

#ifdef _ENABLE_OMP_TARGET_
#include "Impl/CT_OMP_TARGET/CT_OMP_TARGET.h"
#endif

#ifdef _ENABLE_OPENSHMEM_
#include <shmem.h>
#include "Impl/CT_OPENSHMEM/CT_SHMEM.h"
#endif

#ifdef _ENABLE_MPI_
#include <mpi.h>
#include "Impl/CT_MPI/CT_MPI.h"
#endif

#ifdef _ENABLE_XBGAS_
#include <xbrtime.h>
#include "Impl/CT_XBGAS/CT_XBGAS.h"
#endif

#ifdef _ENABLE_PTHREADS_
#include <pthread.h>
#include "Impl/CT_PTHREADS/CT_PTHREADS.h"
#endif

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#include "Impl/CT_OPENACC/CT_OPENACC.h"
#endif

// -----------------------------------------
#ifdef _ENABLE_OPENCL_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include "Impl/CT_OPENCL/CT_OPENCL.h"
#else
#include <CL/cl.h>
#include "Impl/CT_OPENCL/CT_OPENCL.h"
#endif

#include "Impl/CT_OPENCL/CT_OPENCL.h"
#endif
// -----------------------------------------

void PrintTiming( double Timing, double GAMS );

#ifdef _ENABLE_OPENACC_
void RunBenchOpenACC( CTOpts *Opts ){
  // init the OpenACC object
  CT_OPENACC *CT = new CT_OPENACC(Opts->GetBenchType(),
                                  Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_OPENACC OBJECTS" << std::endl;
    return;
  }

  // Set the target options
  if ( !CT->SetDevice() ){
    std::cout << "ERROR : UNABLE TO SET TARGET OPTIONS FOR CT_OPENACC" << std::endl;
    free( CT );
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_OPENACC" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OPENACC" << std::endl;
    CT->FreeData();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OPENACC" << std::endl;
    free( CT );
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // free the structure
  free( CT );
}
#endif

#ifdef _ENABLE_PTHREADS_
void RunBenchPthreads( CTOpts *Opts ){
  // init the PTHREADS object
  CT_PTHREADS *CT = new CT_PTHREADS(Opts->GetBenchType(),
                                    Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_PTHREADS OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_PTHREADS" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_PTHREADS" << std::endl;
    CT->FreeData();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_PTHREADS" << std::endl;
    free( CT );
    return ;
  }

  PrintTiming( Timing, GAMS );
}
#endif

#ifdef _ENABLE_XBGAS_
void RunBenchXBGAS( CTOpts *Opts ){
  // init the XBGAS object
  CT_XBGAS *CT = new CT_XBGAS(Opts->GetBenchType(),
                              Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_XBGAS OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_XBGAS" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_XBGAS" << std::endl;
    CT->FreeData();
    xbrtime_close();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_XBGAS" << std::endl;
    free( CT );
    xbrtime_close();
    return ;
  }

  // Print the timing
  if( xbrtime_mype() == 0 ){
    PrintTiming( Timing, GAMS );
  }
  xbrtime_close();
}
#endif

#ifdef _ENABLE_MPI_
void RunBenchMPI( CTOpts *Opts ){
  // init the MPI object
  CT_MPI *CT = new CT_MPI(Opts->GetBenchType(),
                          Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_MPI OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_MPI" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_MPI" << std::endl;
    CT->FreeData();
    MPI_Finalize();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_MPI" << std::endl;
    MPI_Finalize();
    free( CT );
    return ;
  }

  // Print the timing
  int rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if( rank == 0 ){
    PrintTiming( Timing, GAMS );
  }

  MPI_Finalize();
  free( CT );
}
#endif

#ifdef _ENABLE_OPENSHMEM_
void RunBenchOpenSHMEM( CTOpts *Opts ){
  // init the OpenSHMEM object
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
    shmem_finalize();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_SHMEM" << std::endl;
    shmem_finalize();
    free( CT );
    return ;
  }

  // Print the timing
  if( shmem_my_pe() == 0 ){
    PrintTiming( Timing, GAMS );
  }
  shmem_finalize();
  free( CT );
}
#endif

#ifdef _ENABLE_OMP_TARGET_
void RunBenchOMPTarget( CTOpts *Opts ){
  // init the OpenMP Target object
  CT_OMP_TARGET *CT = new CT_OMP_TARGET(Opts->GetBenchType(),
                                        Opts->GetAtomType());
  if( !CT ){
    std::cout << "ERROR : COULD NOT ALLOCATE CT_OMP_TARGET OBJECTS" << std::endl;
    return ;
  }

  // Set the target options
  if ( !CT->SetDevice() ){
    std::cout << "ERROR : UNABLE TO SET TARGET OPTIONS FOR CT_OMP_TARGET" << std::endl;
    free( CT );
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_OMP_TARGET" << std::endl;
    free( CT );
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OMP_TARGET" << std::endl;
    CT->FreeData();
    free( CT );
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OMP_TARGET" << std::endl;
    free( CT );
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // free the structure
  free( CT );
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

  // free the structure
  free( CT );
}
#endif

// TODO:
#ifdef _ENABLE_OPENCL_
void RunBenchOCL() {
  // TODO:
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
#ifdef _ENABLE_OMP_TARGET_
    RunBenchOMPTarget(Opts);
#endif
#ifdef _ENABLE_OPENSHMEM_
    RunBenchOpenSHMEM(Opts);
#endif
#ifdef _ENABLE_MPI_
    RunBenchMPI(Opts);
#endif
#ifdef _ENABLE_XBGAS_
    RunBenchXBGAS(Opts);
#endif
#ifdef _ENABLE_PTHREADS_
    RunBenchPthreads(Opts);
#endif

#ifdef _ENABLE_OPENACC_
    RunBenchOpenACC(Opts);
#endif

#ifdef _ENABLE_OPENCL_
    RunBenchOCL(Opts);
#endif

  }

  delete Opts;
  return 0;
}

// EOF
