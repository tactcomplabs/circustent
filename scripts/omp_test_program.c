#include <stdio.h>
#include <omp.h>

int main() {
  const int N = 1000;
  int data[N];

  // Initialize data on the host
  for (int i = 0; i < N; ++i) {
    data[i] = 0;
  }

  printf("Starting target offloading...\n");

  // Offload to the target device
  #pragma omp target teams distribute parallel for map(tofrom: data[0:N])
  for (int i = 0; i < N; ++i) {
    data[i] = i;
  }

  printf("Completed target offloading.\n");

  // Verify the results
  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (data[i] != i) {
      printf("Error at index %d: expected %d, got %d\n", i, i, data[i]);
      errors++;
    }
  }

  if (errors == 0) {
    printf("OpenMP target offloading test passed.\n");
  } else {
    printf("OpenMP target offloading test failed with %d errors.\n", errors);
  }

  return 0;
}

