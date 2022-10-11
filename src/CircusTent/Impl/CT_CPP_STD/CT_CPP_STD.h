//
// _CT_CPP_STD_H_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_CPP_STD
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent C++ Standard Atomics Implementation
 *
 */

#ifdef _ENABLE_CPP_STD_

#ifndef _CT_CPP_STD_H_
#define _CT_CPP_STD_H_

#include <cstdlib>
#include <thread>
#include <atomic>
#include <ctime>

#include "CircusTent/CTBaseImpl.h"

class CT_CPP_STD : public CTBaseImpl{
private:
  std::atomic<std::uint64_t> *Array;    ///< CT_CPP_STD: Data array
  std::atomic<std::uint64_t> *Idx;      ///< CT_CPP_STD: Index array
  uint64_t memSize;                     ///< CT_CPP_STD: Memory size (in bytes)
  uint64_t pes;                         ///< CT_CPP_STD: Number of processing elements
  uint64_t iters;                       ///< CT_CPP_STD: Number of iterations per thread
  uint64_t elems;                       ///< CT_CPP_STD: Number of u8 elements
  uint64_t stride;                      ///< CT_CPP_STD: Stride in elements
  uint64_t* expected;                   ///< CT_CPP_STD: Expected Array for CAS kernels

public:
  /// CircusTent C++ standard atomics constructor
  CT_CPP_STD(CTBaseImpl::CTBenchType B,
             CTBaseImpl::CTAtomType A);

  /// CircusTent C++ standard atomics destructor
  ~CT_CPP_STD();

  /// CircusTent C++ standard atomics exeuction function
  virtual bool Execute(double &Timing,double &GAMS) override;

  /// CircusTent C++ standard atomics data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent C++ standard atomics data free function
  virtual bool FreeData() override;

  /// Simple barrier implementation (avoids need for c++20 support)
  /// Note that barrier_ctr must be reset to 0 manually before reuse
  void MyBarrier(std::atomic<std::uint64_t> *barrier_ctr);

  /// Helper function
  void JoinThreads(std::thread *threads);

  /// RAND AMO ADD Benchmark
  void RAND_ADD(uint64_t thread_id,
                std::atomic<std::uint64_t> *barrier_ctr,
                double* start_time);

  /// RAND AMO CAS Benchmark
  void RAND_CAS(uint64_t thread_id,
                std::atomic<std::uint64_t> *barrier_ctr,
                double* start_time);

  /// STRIDE1 AMO ADD Benchmark
  void STRIDE1_ADD(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// STRIDE1 AMO CAS Benchmark
  void STRIDE1_CAS(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// STRIDEN AMO ADD Benchmark
  void STRIDEN_ADD(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// STRIDEN AMO CAS Benchmark
  void STRIDEN_CAS(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// PTRCHASE AMO ADD Benchmark
  void PTRCHASE_ADD(uint64_t thread_id,
                    std::atomic<std::uint64_t> *barrier_ctr,
                    double* start_time);

  /// PTRCHASE AMO CAS Benchmark
  void PTRCHASE_CAS(uint64_t thread_id,
                    std::atomic<std::uint64_t> *barrier_ctr,
                    double* start_time);

  /// SG AMO ADD Benchmark
  void SG_ADD(uint64_t thread_id,
              std::atomic<std::uint64_t> *barrier_ctr,
              double* start_time);

  /// SG AMO CAS Benchmark
  void SG_CAS(uint64_t thread_id,
              std::atomic<std::uint64_t> *barrier_ctr,
              double* start_time);

  /// CENTRAL AMO ADD Benchmark
  void CENTRAL_ADD(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// CENTRAL AMO CAS Benchmark
  void CENTRAL_CAS(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// SCATTER AMO ADD Benchmark
  void SCATTER_ADD(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// SCATTER AMO CAS Benchmark
  void SCATTER_CAS(uint64_t thread_id,
                   std::atomic<std::uint64_t> *barrier_ctr,
                   double* start_time);

  /// GATHER AMO ADD Benchmark
  void GATHER_ADD(uint64_t thread_id,
                  std::atomic<std::uint64_t> *barrier_ctr,
                  double* start_time);

  /// GATHER AMO CAS Benchmark
  void GATHER_CAS(uint64_t thread_id,
                  std::atomic<std::uint64_t> *barrier_ctr,
                  double* start_time);
};

#endif  // _CT_CPP_STD_H_
#endif  // _ENABLE_CPP_STD_

// EOF
