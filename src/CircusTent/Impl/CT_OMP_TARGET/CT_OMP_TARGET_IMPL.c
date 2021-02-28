/*
 * _CT_OMP_TARGET_IMPL_C
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <omp.h>
#include <stdint.h>

/* OpenMP Target Benchmark Implementations
 *
 * Benchmark implementations are in the form:
 *
 * void BENCHTYPE_ATOMTYPE( uint64_t *ARRAY, uint64_t *IDX,
 *                          unsigned long long iters,
 *                          unsigned long long pes )
 *
 */

void RAND_ADD( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               uint64_t iters,
               uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_fetch_add( &ARRAY[IDX[i]], (uint64_t)(0x1), __ATOMIC_RELAXED );
      }
    }
  }
}

void RAND_CAS( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               uint64_t iters,
               uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_compare_exchange_n( &ARRAY[IDX[i]], &ARRAY[IDX[i]], ARRAY[IDX[i]],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_fetch_add( &ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED );
      }
    }
  }
}

void STRIDE1_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes, stride)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads) * stride) );

      for( i=start; i<(start+iters_per_thread); i+=stride ){
        __atomic_fetch_add( &ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED );
      }
    }
  }
}

void STRIDEN_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes, stride)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads) * stride) );

      for( i=start; i<(start+iters_per_thread); i+=stride ){
        __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], ARRAY[i],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void PTRCHASE_ADD( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   uint64_t iters,
                   uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=0; i<iters_per_thread; i++ ){
        start = __atomic_fetch_add( &IDX[start],
                                    (uint64_t)(0x00ull),
                                    __ATOMIC_RELAXED );
      }
    }
  }
}

void PTRCHASE_CAS( uint64_t *restrict ARRAY,
                   uint64_t *restrict IDX,
                   uint64_t iters,
                   uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=0; i<iters_per_thread; i++ ){
        __atomic_compare_exchange_n( &IDX[start], &start, IDX[start],
                                    0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void SG_ADD( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             uint64_t iters,
             uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t src = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        src  = __atomic_fetch_add( &IDX[i], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
        dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[src], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
      }
    }
  }
}

void SG_CAS( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             uint64_t iters,
             uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t src = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
      __atomic_compare_exchange_n( &IDX[i], &src, IDX[i],
                                   0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                   0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      __atomic_compare_exchange_n( &ARRAY[src], &val, ARRAY[src],
                                   0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      __atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
                                   0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);

      for( i=0; i<iters_per_thread; i++ ){
        __atomic_fetch_add( &ARRAY[0], (uint64_t)(0x1), __ATOMIC_RELAXED );
      }
    }
  }
}

void CENTRAL_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);

      for( i=0; i<iters_per_thread; i++ ){
        __atomic_compare_exchange_n( &ARRAY[0], &ARRAY[0], ARRAY[0],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[i], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[dest], val, __ATOMIC_RELAXED );
      }
    }
  }
}

void SCATTER_CAS( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[i], &val, ARRAY[i],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[dest], &ARRAY[dest], val,
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

void GATHER_ADD( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );

      for( i=start; i<(start+iters_per_thread); i++ ){
        dest = __atomic_fetch_add( &IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED );
        val = __atomic_fetch_add( &ARRAY[dest], (uint64_t)(0x01ull), __ATOMIC_RELAXED );
        __atomic_fetch_add( &ARRAY[i], val, __ATOMIC_RELAXED );
      }
    }
  }
}

void GATHER_CAS( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes ){

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, pes)
  {
    #pragma omp parallel
    {
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;

      // Divide iters across number of threads per team & set start
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (omp_get_thread_num() == num_threads - 1) ?
                                  (iters / num_threads) + (iters % num_threads) :
                                  (iters / num_threads);
      uint64_t start = (uint64_t) ( (omp_get_team_num() * iters) +
                                    (omp_get_thread_num() * (iters/num_threads)) );


      for( i=start; i<(start+iters_per_thread); i++ ){
        __atomic_compare_exchange_n( &IDX[i+1], &dest, IDX[i+1],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[dest], &val, ARRAY[dest],
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
        __atomic_compare_exchange_n( &ARRAY[i], &ARRAY[i], val,
                                     0, __ATOMIC_RELAXED, __ATOMIC_RELAXED );
      }
    }
  }
}

/* EOF */
