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

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread); i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[IDX[i]];
          ARRAY[IDX[i]] += 1;
        }
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

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread); i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += 1;
        }
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
      uint64_t num_teams = (uint64_t) omp_get_num_teams();
      uint64_t num_threads = (uint64_t) omp_get_num_threads();
      uint64_t iters_per_thread = (uint64_t) ( ( (omp_get_thread_num()) == (num_threads-1) ) ?
                                               ( (iters/num_threads) + (iters%num_threads) )
                                               (iters/num_threads) );
      uint64_t start= (omp_get_team_num() * iters * stride) + (omp_get_thread_num() * (iters/num_threads) * stride);

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread*stride); i+=stride ){
        #pragma omp atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += 1;
        }
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
        #pragma omp atomic capture
        {
          start = IDX[start];
          IDX[start] += 0;
        }
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

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread); i++ ){
        #pragma omp atomic capture
        {
          src = IDX[i];
          IDX[i] += 0;
        }

        #pragma omp atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += 0;
        }

        #pragma omp atomic capture
        {
          val = ARRAY[src];
          ARRAY[src] += 1;
        }

        #pragma omp atomic capture
        {
          ret = ARRAY[dest];
          ARRAY[dest] += val;
        }
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

      uint64_t ret;
      for( i=0; i<iters_per_thread; i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[0];
          ARRAY[0] += 1;
        }
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

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread); i++ ){
        #pragma omp atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += 0;
        }

        #pragma omp atomic capture
        {
          val = ARRAY[i];
          ARRAY[i] += 1;
        }

        #pragma omp atomic capture
        {
          ret = ARRAY[dest];
          ARRAY[dest] += val;
        }
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

      uint64_t ret;
      for( i=start; i<(start+iters_per_thread); i++ ){
        #pragma omp atomic capture
      	{
          dest = IDX[i+1];
          IDX[i+1] += 0;
        }

        #pragma omp atomic capture
        {
          val = ARRAY[dest];
          ARRAY[dest] += 1;
        }

        #pragma omp atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += val;
        }
      }
    }
  }
}

/* EOF */
