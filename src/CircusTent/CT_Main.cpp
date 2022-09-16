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
#include "Impl/CT_OPENSHMEM/CT_SHMEM.h"
#endif

#ifdef _ENABLE_MPI_
#include "Impl/CT_MPI/CT_MPI.h"
#endif

#ifdef _ENABLE_XBGAS_
#include "Impl/CT_XBGAS/CT_XBGAS.h"
#endif

#ifdef _ENABLE_PTHREADS_
#include "Impl/CT_PTHREADS/CT_PTHREADS.h"
#endif

#ifdef _ENABLE_OPENACC_
#include "Impl/CT_OPENACC/CT_OPENACC.h"
#endif

#ifdef _ENABLE_OPENCL_
#include "Impl/CT_OPENCL/CT_OPENCL.h"
#endif

#ifdef _ENABLE_CPP_STD_
#include "Impl/CT_CPP_STD/CT_CPP_STD.h"
#endif

#ifdef _ENABLE_CUDA_
#include "Impl/CT_CUDA/CT_CUDA.cuh"
#endif

void PrintTiming( double Timing, double GAMS );

#ifdef _ENABLE_CUDA_
void RunBenchCuda(CTOpts *Opts) {

  // Init the CUDA object
  CT_CUDA *CT = new CT_CUDA(
    Opts->GetBenchType(),
    Opts->GetAtomType()
  );

  if ( !CT ) {
    std::cout << "ERROR: COULD NOT ALLOCATE CT_CUDA OBJECTS" << std::endl;
    return;
  }

  // Take in options for blocksPerGrid and threadsPerBlock
  if ( !CT->ParseCUDAOpts( Opts->GetArgc(), Opts->GetArgv() ) ) {
    std::cout << "Failed to parse CUDA command line options" << std::endl;
    delete CT;
    return ;
  }

  // TODO: Set the target options

  // Print device information
  if ( !CT->PrintCUDADeviceProperties( CT->GetCUDAdeviceID(), CT->GetCUDAdeviceCount() ) ) {
    std::cout << "ERROR: COULD NOT PRINT CUDA DEVICE PROPERTIES FOR CT_CUDA" << std::endl;
    delete CT;
    return;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_CUDA" << std::endl;
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS   = 0.;

  if ( !CT->Execute(Timing, GAMS) ) {
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_CUDA" << std::endl;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // Free the data
  if ( !CT->FreeData() ) {
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_CUDA" << std::endl;
    delete CT;
    return ;
  }

  // Free the structure
  delete CT;
}

#endif

#ifdef _ENABLE_CPP_STD_
void RunBenchCppStd(CTOpts *Opts) {
  // Init the object
  CT_CPP_STD *CT = new CT_CPP_STD(Opts->GetBenchType(),
                          	     Opts->GetAtomType());

  if ( !CT ) {
    std::cout << "ERROR: COULD NOT ALLOCATE CT_CPP_STD OBJECTS" << std::endl;
    return ;
  }

  // Allocate the data
  if (!CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride())){
      std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR CT_CPP_STD" << std::endl;
      CT->FreeData();
      delete CT;
      return;
  }

  // Execute the benchmark
  double Timing = 0;
  double GAMS = 0.;
  if ( !CT->Execute(Timing, GAMS) ) {
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_CPP_STD" << std::endl;
    CT->FreeData();
    free ( CT );
    return;
  }

  // Free the data
  if ( !CT->FreeData() ) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR CT_OCL" << std::endl;
    delete CT;
    return;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // Free the structure
  delete CT;
}
#endif

#ifdef _ENABLE_OPENCL_
void RunBenchOpenCL(CTOpts *Opts) {
  // Init the OpenCL Object
  CT_OPENCL *CT = new CT_OPENCL(Opts->GetBenchType(),
                          	Opts->GetAtomType());

  if ( !CT ) {
    std::cout << "ERROR: COULD NOT ALLOCATE CT_OCL OBJECTS" << std::endl;
    return ;
  }

  // Initialize the OCL environment
  if ( !CT->Initialize() ){
    std::cout << "ERROR : COULD NOT INITIALIZE CT_OCL ENVIRONMENT" << std::endl;
    CT->FreeData();
    delete CT;
    return ;
  }

  // Allocate the data
  if (!CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride())){
      std::cout << "ERROR: COULD NOT ALLOCATE MEMORY FOR CT_OCL" << std::endl;
      CT->FreeData();
      delete CT;
      return;
  }

  // Execute the benchmark
  double Timing = 0;
  double GAMS = 0.;
  if ( !CT->Execute(Timing, GAMS) ) {
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OCL" << std::endl;
    CT->FreeData();
    free ( CT );
    return;
  }

  // Free the data
  if ( !CT->FreeData() ) {
    std::cout << "ERROR: COULD NOT FREE THE MEMORY FOR CT_OCL" << std::endl;
    delete CT;
    return;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // Free the structure
  delete CT;
}
#endif

#ifdef _ENABLE_OPENACC_
void RunBenchOpenACC( CTOpts *Opts ){

  if(Opts->GetAtomType() == CTBaseImpl::CTAtomType::CT_CAS){
    std::cout << "ERROR : CAS IMPLEMENTATION NOT SUPPORTED IN CT_OPENACC" << std::endl;
    return;
  }

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
    delete CT;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_OPENACC" << std::endl;
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OPENACC" << std::endl;
    CT->FreeData();
    delete CT;
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OPENACC" << std::endl;
    delete CT;
    return ;
  }

  // free the structure
  delete CT;
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
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_PTHREADS" << std::endl;
    CT->FreeData();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_PTHREADS" << std::endl;
    delete CT;
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
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_XBGAS" << std::endl;
    CT->FreeData();
    xbrtime_close();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_XBGAS" << std::endl;
    delete CT;
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
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_MPI" << std::endl;
    CT->FreeData();
    MPI_Finalize();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_MPI" << std::endl;
    MPI_Finalize();
    delete CT;
    return ;
  }

  // Print the timing
  int rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if( rank == 0 ){
    PrintTiming( Timing, GAMS );
  }

  MPI_Finalize();
  delete CT;
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
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_SHMEM" << std::endl;
    CT->FreeData();
    shmem_finalize();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_SHMEM" << std::endl;
    shmem_finalize();
    delete CT;
    return ;
  }

  // Print the timing
  if( shmem_my_pe() == 0 ){
    PrintTiming( Timing, GAMS );
  }
  shmem_finalize();
  delete CT;
}
#endif

#ifdef _ENABLE_OMP_TARGET_
void RunBenchOMPTarget( CTOpts *Opts ){

  if(Opts->GetAtomType() == CTBaseImpl::CTAtomType::CT_CAS){
    std::cout << "ERROR : CAS IMPLEMENTATION NOT SUPPORTED IN CT_OMP_TARGET" << std::endl;
    return;
  }

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
    delete CT;
    return ;
  }

  // Allocate the data
  if( !CT->AllocateData( Opts->GetMemSize(),
                         Opts->GetPEs(),
                         Opts->GetIters(),
                         Opts->GetStride() ) ){
    std::cout << "ERROR : COULD NOT ALLOCATE MEMORY FOR CT_OMP_TARGET" << std::endl;
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OMP_TARGET" << std::endl;
    CT->FreeData();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OMP_TARGET" << std::endl;
    delete CT;
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // free the structure
  delete CT;
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
    delete CT;
    return ;
  }

  // Execute the benchmark
  double Timing = 0.;
  double GAMS = 0.;
  if( !CT->Execute(Timing,GAMS) ){
    std::cout << "ERROR : COULD NOT EXECUTE BENCHMARK FOR CT_OMP" << std::endl;
    CT->FreeData();
    delete CT;
    return ;
  }

  // Free the data
  if( !CT->FreeData() ){
    std::cout << "ERROR : COULD NOT FREE THE MEMORY FOR CT_OMP" << std::endl;
    delete CT;
    return ;
  }

  // Print the timing
  PrintTiming( Timing, GAMS );

  // free the structure
  delete CT;
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

  if( (!Opts->IsHelp()) && (!Opts->IsList()) )3{
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
    RunBenchOpenCL(Opts);
#endif
#ifdef _ENABLE_CPP_STD_
    RunBenchCppStd(Opts);
#endif
#ifdef _ENABLE_CUDA_
    RunBenchCuda(Opts);
#endif
  }

  delete Opts;
  return 0;
}

// EOF
