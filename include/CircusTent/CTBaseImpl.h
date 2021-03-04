//
// _CTBaseImpl_h_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CTBaseImpl
 *
 * \ingroup CircusTent
 *
 * \brief Base implementation template class
 *
 */

#ifndef _CTBASEIMPL_H_
#define _CTBASEIMPL_H_

#include <iostream>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <sys/time.h>

class CTBaseImpl{
public:
  /// CTBaseImpl: Benchmark types
  typedef enum{
    CT_NB         = 0,    ///< CTBenchType: Null benchmark type
    CT_RAND       = 1,    ///< CTBenchType: Random access test
    CT_STRIDE1    = 2,    ///< CTBenchType: Stride 1 access test
    CT_STRIDEN    = 3,    ///< CTBenchType: Stride by N elems access test
    CT_PTRCHASE   = 4,    ///< CTBenchType: Pointer chase
    CT_SG         = 5,    ///< CTBenchType: Scatter/Gather
    CT_CENTRAL    = 6,    ///< CTBenchType: Centralized access (one point)
    CT_SCATTER    = 7,    ///< CTBenchType: Scatter
    CT_GATHER     = 8     ///< CTBenchType: Gather
  }CTBenchType;

  /// CTBaseImpl: Atomic operation types
  typedef enum{
    CT_NA         = 0,    ///< CTAtomType: Null atomic type
    CT_ADD        = 1,    ///< CTAtomType: Atomic Add
    CT_CAS        = 2     ///< CTAtomType: Compare and Swap
  }CTAtomType;

  /// Default constructor
  CTBaseImpl(std::string N,
             CTBaseImpl::CTBenchType B,
             CTBaseImpl::CTAtomType A) : Impl(N),
                                         BenchType(B),
                                         AtomType(A) {}

  /// Default virtual destructor
  virtual ~CTBaseImpl() {}

  /// Virtual execution function
  virtual bool Execute(double &Timing,double &GAMS) = 0;

  /// Virtual data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) = 0;

  /// Virtual data free function
  virtual bool FreeData() = 0;

  /// Retrieves the name of the implementation
  std::string GetImplName() { return Impl; }

  /// Retrieves the benchmark type
  CTBaseImpl::CTBenchType GetBenchType() { return BenchType; }

  /// Retrieves the atomic type
  CTBaseImpl::CTAtomType GetAtomType() { return AtomType; }

  /// Determines if the bench type is valid
  bool IsValidBench( CTBaseImpl::CTBenchType BenchType ){
    if( BenchType > 8 )
      return false;
    return true;
  }

  /// Determines if the atomic type is valid
  bool IsValidAtomic( CTBaseImpl::CTAtomType  AtomType ){
    if( AtomType > 2 )
      return false;
    return true;
  }

  /// Reports an error with a benchmark combination doesn't exist
  void ReportBenchError() {
    std::cout << "ERROR : BENCHMARK COMBINATION DOES NOT EXIST" << std::endl;
  }

  /// Reports the current cpu second timer
  double MySecond() {
    struct timeval tp;
    struct timezone tzp;

    gettimeofday( &tp, &tzp );
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
  }

  /// Reports the runtime of a benchmark
  double Runtime( double StartTime, double EndTime ){
    return EndTime - StartTime;
  }

  /// Reports the number of Giga atomic operations
  double GAM(double Ops, double Iters, double PEs){
    return (Ops*Iters*PEs)/1000000000.0;
  }

private:
  std::string Impl;                   ///< Implementation name
  CTBaseImpl::CTBenchType BenchType;  ///< Benchmark Type
  CTBaseImpl::CTAtomType AtomType;    ///< Atomic Type
};

#endif

// EOF
