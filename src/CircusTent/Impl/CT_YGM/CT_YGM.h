//
// _CT_YGM_H_
//
// Copyright (C) 2017-2023 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

/**
 * \class CT_YGM
 *
 * \ingroup CircusTent
 *
 * \brief CircusTent YGM Implementation
 *
 */

#ifdef _ENABLE_YGM_

#ifndef _CT_YGM_H_
#define _CT_YGM_H_

#include <cstdlib>
#include <random>
#include <ctime>
#include <ygm/comm.hpp>

#include "CircusTent/CTBaseImpl.h"

class CT_YGM : public CTBaseImpl{
private:

  // In YGM, there is only one thread per pe, so
  // std::atomic wrapper for VAL and IDX arrays
  // as seen in the CPP STD implementation
  // is not necessary.

  // Additionally, if VAL and IDX are static
  // we do not need to use YGM pointers to access them on
  // remote ranks and can save on message space

  static uint64_t* val;          ///< CT_YGM: VAL array
  static uint64_t* idx;          ///< CT_YGM: IDX array
  
  int target;                    ///< CT_YGM: target remote pe for benchmarks where necessary

  uint64_t memSize;              ///< CT_YGM: Memory size (in bytes)
  uint64_t pes;                  ///< CT_YGM: Number of processing elements
  static uint64_t iters;         ///< CT_YGM: Number of iterations per thread
  uint64_t elems;                ///< CT_YGM: Number of u8 elements stored in Array at each rank
  uint64_t stride;               ///< CT_YGM: Stride in elements
  uint64_t rank;                 ///< CT_YGM: Rank of the current PE
  uint64_t chasers_per_rank;     ///< CT_YGM: Number of PTRCHASE functors to start from each PE

  ygm::comm world;               ///< CT_YGM: Communicator for YGM runtime

  // -- private member implementations for YGM
  // RAND AMO ADD Benchmark
  void RAND_ADD();

  /// RAND AMO CAS Benchmark
  void RAND_CAS();

  /// STRIDE1 AMO ADD Benchmark
  void STRIDE1_ADD();

  /// STRIDE1 AMO CAS Benchmark
  void STRIDE1_CAS();

  /// STRIDEN AMO ADD Benchmark
  void STRIDEN_ADD();

  /// STRIDEN AMO CAS Benchmark
  void STRIDEN_CAS();

  /// CENTRAL AMO ADD Benchmark
  void CENTRAL_ADD();

  /// CENTRAL AMO CAS Benchmark
  void CENTRAL_CAS();

  /// PTRCHASE AMO ADD Benchmark
  void PTRCHASE_ADD();

  /// PTRCHASE AMO CAS Benchmark
  void PTRCHASE_CAS();

  /// SCATTER AMO ADD Benchmark
  void SCATTER_ADD();

  /// SCATTER AMO CAS Benchmark
  void SCATTER_CAS();

  /// GATHER AMO ADD Benchmark
  void GATHER_ADD();

  /// GATHER AMO CAS Benchmark
  void GATHER_CAS();

  /// SG AMO ADD Benchmark
  void SG_ADD();

  /// SG AMO CAS Benchmark
  void SG_CAS();

  // -- private functors that allow recursive function calls as needed for specific benchmarks
  // PTRCHASE AMO ADD Functor
  struct chase_functor_add {
  public:

    // for i ← 0 to iters by 1 do
    //     start = AMO(IDX[start])
    // end

    template <typename Comm>
    void operator()(Comm* pcomm, uint64_t index, uint64_t value, uint64_t ops_left) {

      // AMO(IDX[start])
      idx[index] += value;

      // index = IDX[start]
      index = idx[index];

      ops_left--;

      // continue recursive chasing until performed all hops
      if (ops_left > 0) {
        pcomm->async(index/(iters + 1), chase_functor_add(), index % (iters + 1), value, ops_left);

#ifdef _PROGRESS_PTRCHASE_
        pcomm->local_progress();
#endif

      }
    }
  };

  // PTRCHASE AMO CAS Functor
  struct chase_functor_cas {
  public:

    // for i ← 0 to iters by 1 do
    //     start = AMO(IDX[start])
    // end

    template <typename Comm>
    void operator()(Comm* pcomm, uint64_t index, uint64_t desired, uint64_t ops_left) {

      // CAS behavior similar to CPP STD implementation
      desired = idx[index];

      // CAS(IDX[start])
      if ((idx[index] % (iters + 1)) == index)
      { 
        idx[index] = desired;
      }

      // start = IDX[start]
      index = idx[index];

      ops_left--;

      // continue recursive chasing until performed all hops
      if (ops_left > 0) {
        pcomm->async(index/(iters + 1), chase_functor_cas(), index % (iters + 1), desired, ops_left);

#ifdef _PROGRESS_PTRCHASE_
        pcomm->local_progress();
#endif

      }
    }
  };

public:
  /// CircusTent YGM Constructor
  CT_YGM(CTBaseImpl::CTBenchType B,
         CTBaseImpl::CTAtomType A);

  /// CircusTent YGM destructor
  ~CT_YGM();

  /// CircusTent YGM execution function
  virtual bool Execute(double &Timing, double &GAMS) override;

  /// CircusTent YGM data allocation function
  virtual bool AllocateData( uint64_t memSize,
                             uint64_t pes,
                             uint64_t iters,
                             uint64_t stride ) override;

  /// CircusTent YGM data free function
  virtual bool FreeData() override;

  // Debug function for checking Array contents
  void PrintVal();

  // Debug function for checking Idx contents
  void PrintIdx();
};

#endif  // _CT_YGM_H_
#endif  // _ENABLE_YGM_

// EOF
