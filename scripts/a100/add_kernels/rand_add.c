#include <stdio.h>
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
        // Use ret to avoid the warning
        if (i % (iters / 10) == 0) {
          printf("Team %d: ARRAY[IDX[%lu]] was %lu\n", omp_get_team_num(), i, ret);
        }
      }

      printf("Team %d completed...\n", omp_get_team_num());
      fflush(stdout);
    }
  }

  printf("Completed RAND_ADD kernel.\n");
  fflush(stdout);
}

int main() {
  // For testing purposes, create a simple main function to run RAND_ADD
  uint64_t size = 1024;
  uint64_t array[size];
  uint64_t idx[size];
  uint64_t iters = 1000;
  uint64_t pes = 16;

  // Initialize arrays
  for (uint64_t i = 0; i < size; ++i) {
    array[i] = i;
    idx[i] = i % size;
  }

  RAND_ADD(array, idx, iters, pes);

  return 0;
}

