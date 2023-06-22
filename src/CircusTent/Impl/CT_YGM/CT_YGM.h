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

  // In YGM, there is only one thread per pe, so
  // std::atomic wrapper for VAL and IDX arrays
  // as seen in the CPP STD implementation
  // is not necessary.
  uint64_t *Array;               ///< CT_YGM: VAL array
  uint64_t *Idx;                 ///< CT_YGM: IDX array
  
  int target;                    ///< CT_YGM: target remote pe for benchmarks where necessary

  uint64_t memSize;              ///< CT_YGM: Memory size (in bytes)
  uint64_t pes;                  ///< CT_YGM: Number of processing elements
  uint64_t iters;                ///< CT_YGM: Number of iterations per thread
  uint64_t elems;                ///< CT_YGM: Number of u8 elements stored in Array at each rank
  uint64_t stride;               ///< CT_YGM: Stride in elements
  uint64_t rank;                 ///< CT_YGM: Rank of the current PE

  typename ygm::ygm_ptr<uint64_t*> yp_Array;    ///< CT_YGM: YGM VAL ygm pointer
  typename ygm::ygm_ptr<uint64_t*> yp_Idx;      ///< CT_YGM: YGM IDX ygm pointer

  ygm::comm world;               ///< CT_YGM: Communicator for YGM benchmarks

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

  void SCATTER_ADD_ALTERNATE();

  /// SCATTER AMO CAS Benchmark
  void SCATTER_CAS();

  void SCATTER_CAS_ALTERNATE();

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
    void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t index, uint64_t value, uint64_t ops_left, uint64_t block_size) {

      // AMO(IDX[start])
      (*parray)[index] += value;

      // index = IDX[start]
      index = (*parray)[index];

      ops_left--;

      // continue recursive chasing until performed all hops
      if (ops_left > 0) {
        pcomm->async(index/block_size, chase_functor_add(), parray, index % block_size, value, ops_left, block_size);
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
      void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t index, uint64_t desired, uint64_t ops_left, uint64_t block_size) {

        // CAS behavior similar to CPP STD implementation
        desired = (*parray)[index];

        // CAS(IDX[start])
        if (((*parray)[index] % block_size) == index)
        { 
            (*parray)[index] = desired;
        }

        // start = IDX[start]
        index = (*parray)[index];

        ops_left--;

        // continue recursive chasing until performed all hops
        if (ops_left > 0) {
          pcomm->async(index/block_size, chase_functor_cas(), parray, index % block_size, desired, ops_left, block_size);
        }
      }
  };

  // GATHER AMO ADD Functor
  struct gather_functor_add {
    public:
      template <typename Comm>
      void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t index, uint64_t iter, uint64_t sender)
      {
        // performs AMO(VAL[i], val) at origin rank
        auto sender_amo = [](auto parray, uint64_t index, uint64_t value)
        {
          (*parray)[index] += value;
        };

        // val = AMO(VAL[src])
        uint64_t val  = (*parray)[index] + (uint64_t)(0x1);
      
        // send call for AMO(VAL[i], val) back to sender using val found here
        pcomm->async(sender, sender_amo, parray, iter, val);
      }
  };

  // GATHER AMO CAS Functor
  struct gather_functor_cas {
    public:
      template <typename Comm>
      void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t index, uint64_t iter, uint64_t sender) 
      {

        uint64_t val = 0x0;

        // CAS for val
        if( (*parray)[index] == 0 ){
          (*parray)[index] = 0;
        }
        val = (*parray)[index];

        auto sender_amo = [](auto parray, uint64_t index, uint64_t value)
        {
          // performs CAS AMO(VAL[i], val) at origin rank
          if( (*parray)[index] == value ){
            // use of zero for desired is like MPI
            (*parray)[index] = 0;
          }
        };
      
        // send call for AMO(VAL[i], val) back to sender using val found here
        pcomm->async(sender, sender_amo, parray, iter, val);
      }
  };

  // SG AMO ADD Functor
  struct sg_functor_add {
    public:
      template <typename Comm>
      void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t val_index, uint64_t reciever, uint64_t amo_index)
      {
        // performs CAS AMO(VAL[dest], val)
        auto reciever_amo = [](auto parray, uint64_t index, uint64_t value)
        {
          (*parray)[index] += value;
        };

        // val = AMO(VAL[src])
        uint64_t val = (*parray)[val_index] + (uint64_t)(0x0);
      
        // now that we know val,
        // send a call for AMO(VAL[dest], val) to owner of VAL[dest]
        pcomm->async(reciever, reciever_amo, parray, amo_index, val);
      }
  };

  // SG AMO CAS Functor
  struct sg_functor_cas {
    public:
      template <typename Comm>
      void operator()(Comm* pcomm, ygm::ygm_ptr<uint64_t*> parray, uint64_t val_index, uint64_t reciever, uint64_t amo_index)
      {
        auto reciever_amo = [](auto parray, uint64_t index, uint64_t value)
        {
          // CAS at VAL[dest] with VAL[src]
          if( (*parray)[index] == value ){
            // again, swap with zero is similar to MPI implementation
            (*parray)[index] = 0;
          }
        };

        uint64_t val = 0x0;

        // val = AMO(VAL[src])
        if( (*parray)[val_index] == 0 ){
          (*parray)[val_index] = 0;
        }
        val = (*parray)[val_index];

        // send a call for AMO(VAL[dest], val) to owner of VAL[dest]
        pcomm->async(reciever, reciever_amo, parray, amo_index, val);
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
  void PrintArray();

  // Debug function for checking Idx contents
  void PrintIdx();
};

#endif  // _CT_YGM_H_
#endif  // _ENABLE_YGM_

// EOF
