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
#include <ctime>
#include <ygm/comm.hpp>
#include <ygm/detail/ygm_ptr.hpp>

#include "CircusTent/CTBaseImpl.h"

class CT_YGM : public CTBaseImpl{
private:

  uint64_t *Array;               ///< CT_YGM: VAL array
  uint64_t *Idx;                 ///< CT_YGM: IDX array
  int *Target;                   ///< CT_YGM: Target? PE array (Consider if necessary for YGM.)

  uint64_t memSize;              ///< CT_YGM: Memory size (in bytes)
  uint64_t pes;                  ///< CT_YGM: Number of processing elements
  uint64_t iters;                ///< CT_YGM: Number of iterations per thread
  uint64_t elems;                ///< CT_YGM: Number of u8 elements
  uint64_t stride;               ///< CT_YGM: Stride in elements

  typename ygm::ygm_ptr<uint64_t*> yp_Array;   ///< CT_YGM: YGM VAL ygm pointer
  typename ygm::ygm_ptr<uint64_t*> yp_Idx;     ///< CT_YGM: YGM IDX ygm pointer

  ygm::comm world = ygm::comm(NULL, NULL);     ///< CT_YGM: Communicator for YGM benchmarks

  // -- private member implementations for YGM
  // RAND AMO ADD Benchmark
  void RAND_ADD( uint64_t iters, uint64_t pes );

  /// STRIDE1 AMO ADD Benchmark
  void STRIDE1_ADD();

  /// STRIDE1 AMO CAS Benchmark
  void STRIDE1_CAS();

  /// STRIDEN AMO ADD Benchmark
  void STRIDEN_ADD();

  /// STRIDEN AMO CAS Benchmark
  void STRIDEN_CAS();

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
};

#endif  // _CT_YGM_H_
#endif  // _ENABLE_YGM_

// EOF
