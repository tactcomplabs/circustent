/*
 * _CT_OPENACC_IMPL_C
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <openacc.h>
#include <stdint.h>

/* OpenACC Benchmark Implementations
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      uint64_t i = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t start = gangID * iters;

      uint64_t ret;
      for( i=start; i<(start+iters); i++ ){
        #pragma acc atomic capture
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      uint64_t i = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t start = gangID * iters;

      uint64_t ret;
      for( i=start; i<(start+iters); i++ ){
        #pragma acc atomic capture
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes, stride)
  {
    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      uint64_t i = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t start = gangID * iters * stride;

      uint64_t ret;
      for( i=start; i<(start+iters*stride); i+=stride ){
        #pragma acc atomic capture
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
  
  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    /* Avoids invalid atomic exprssion *
     * with some compilers for += 0    */
    uint64_t zero = 0;

    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      uint64_t i = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t start = gangID * iters;

      for( i=0; i<iters; i++ ){
        #pragma acc atomic capture
        {
          start = IDX[start];
          IDX[start] += zero;
        }
      }
    }
  }
}

void SG_ADD( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             uint64_t iters,
             uint64_t pes ){

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    /* Avoids invalid atomic exprssion *
     * with some compilers for += 0    */
    uint64_t zero = 0;

    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t i = 0;
      uint64_t src = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t start = gangID * iters;

      uint64_t ret;
      for( i=start; i<(start+iters); i++ ){
        #pragma acc atomic capture
        {
          src = IDX[i];
          IDX[i] += zero;
        }

        #pragma acc atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
        }

        #pragma acc atomic capture
        {
          val = ARRAY[src];
          ARRAY[src] += 1;
        }

        #pragma acc atomic capture
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    #pragma acc parallel num_gangs(pes)
    {
      uint64_t i, ret;
      for( i=0; i<iters; i++ ){
        #pragma acc atomic capture
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      /* Avoids invalid atomic exprssion *
       * with some compilers for += 0    */
      uint64_t zero = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t start = gangID * iters;

      uint64_t ret;
      for( i=start; i<(start+iters); i++ ){
        #pragma acc atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
        }

        #pragma acc atomic capture
        {
          val = ARRAY[i];
          ARRAY[i] += 1;
        }

        #pragma acc atomic capture
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

  #pragma acc data deviceptr(ARRAY, IDX) copyin(iters, pes)
  {
    // target global variable for assigning gang IDs
    uint64_t gangCtr = 0;
    #pragma acc parallel num_gangs(pes)
    {
      /* Avoids invalid atomic exprssion *
       * with some compilers for += 0    */
      uint64_t zero = 0;

      // Atomic F&A to order gangs
      uint64_t gangID;
      #pragma acc atomic capture
      {
        gangID = gangCtr;
        gangCtr++;
      }
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t start = gangID * iters;

      uint64_t ret;
      for( i=start; i<(start+iters); i++ ){
        #pragma acc atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
        }

        #pragma acc atomic capture
        {
          val = ARRAY[dest];
          ARRAY[dest] += 1;
        }

        #pragma acc atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += val;
        }
      }
    }
  }
}

/* EOF */
