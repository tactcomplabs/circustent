#include <omp.h>
#include <stdint.h>
#include <stdio.h>

void RAND_ADD( uint64_t *restrict ARRAY,
               uint64_t *restrict IDX,
               uint64_t iters,
               uint64_t pes ){

  printf("Starting RAND_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t i = 0, ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters);

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+iters); i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[IDX[i]];
          ARRAY[IDX[i]] += 1;
        }
      }

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed RAND_ADD kernel.\n");
  fflush(stdout);
}

void STRIDE1_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  printf("Starting STRIDE1_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t i = 0, ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters);
      
      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+iters); i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += 1;
        }
      }

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed STRIDE1_ADD kernel.\n");
  fflush(stdout);
}

void STRIDEN_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes,
                  uint64_t stride ){

  printf("Starting STRIDEN_ADD kernel with %lu iterations, %lu PEs, and stride %lu...\n", iters, pes, stride);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters, stride)
  {
    #pragma omp parallel
    {
      uint64_t i = 0, ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters * stride);

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+(iters*stride)); i+=stride ){
        #pragma omp atomic capture
        {
          ret = ARRAY[i];
          ARRAY[i] += 1;
        }
      }

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed STRIDEN_ADD kernel.\n");
  fflush(stdout);
}

void SG_ADD( uint64_t *restrict ARRAY,
             uint64_t *restrict IDX,
             uint64_t iters,
             uint64_t pes ){

  printf("Starting SG_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t zero = 0;
      uint64_t i = 0;
      uint64_t src = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters);

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+iters); i++ ){
        #pragma omp atomic capture
        {
          src = IDX[i];
          IDX[i] += zero;
        }

        #pragma omp atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
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

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed SG_ADD kernel.\n");
  fflush(stdout);
}

void CENTRAL_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  printf("Starting CENTRAL_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t i = 0, ret;

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=0; i<iters; i++ ){
        #pragma omp atomic capture
        {
          ret = ARRAY[0];
          ARRAY[0] += 1;
        }
      }

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed CENTRAL_ADD kernel.\n");
  fflush(stdout);
}

void SCATTER_ADD( uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes ){

  printf("Starting SCATTER_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t zero = 0;
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters);

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+iters); i++ ){
        #pragma omp atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
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

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed SCATTER_ADD kernel.\n");
  fflush(stdout);
}

void GATHER_ADD( uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes ){

  printf("Starting GATHER_ADD kernel with %lu iterations and %lu PEs...\n", iters, pes);
  fflush(stdout);

  #pragma omp target teams num_teams(pes) is_device_ptr(ARRAY, IDX) map(to:iters)
  {
    #pragma omp parallel
    {
      uint64_t zero = 0;
      uint64_t i = 0;
      uint64_t dest = 0;
      uint64_t val = 0;
      uint64_t ret;
      uint64_t start = (uint64_t) (omp_get_team_num() * iters);

      printf("Team %d starting...\n", omp_get_team_num());
      fflush(stdout);

      #pragma omp for simd
      for( i=start; i<(start+iters); i++ ){
        #pragma omp atomic capture
        {
          dest = IDX[i+1];
          IDX[i+1] += zero;
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

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed GATHER_ADD kernel.\n");
  fflush(stdout);
}

