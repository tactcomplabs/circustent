/*
 * _CT_PTHREADS_IMPL_C_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include <pthread.h>
#include <stdint.h>


// Struct holding unique function arguments for each thread
typedef struct
{
	uint64_t thread_id;
  uint64_t *ARRAY;
  uint64_t *IDX;
  uint64_t iters;
  uint64_t stride;
}arg_struct;

/* Pthreads Benchmark Implementations
 *
 * Benchmark implementations are in the form:
 *
 * void BENCHTYPE_ATOMTYPE( uint64_t *ARRAY, uint64_t *IDX,
 *                          unsigned long long iters,
 *                          unsigned long long pes )
 *
 */

// Pthreads RAND_ADD helper function
void *thread_RAND_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    __atomic_fetch_add(&ARRAY[IDX[i]], (uint64_t)(0x1), __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void RAND_ADD(uint64_t *restrict ARRAY,
              uint64_t *restrict IDX,
              uint64_t iters,
              uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_RAND_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads RAND_CAS helper function
void *thread_RAND_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&ARRAY[IDX[i]], &ARRAY[IDX[i]], ARRAY[IDX[i]], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void RAND_CAS(uint64_t *restrict ARRAY,
              uint64_t *restrict IDX,
              uint64_t iters,
              uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_RAND_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads STRIDE1_ADD helper function
void *thread_STRIDE1_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    __atomic_fetch_add(&ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void STRIDE1_ADD(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_STRIDE1_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads STRIDE1_CAS helper function
void *thread_STRIDE1_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&ARRAY[i], &ARRAY[i], ARRAY[i], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void STRIDE1_CAS(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_STRIDE1_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads STRIDEN_ADD helper function
void *thread_STRIDEN_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;
  uint64_t stride = arg_ptr->stride;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i += stride){
    __atomic_fetch_add(&ARRAY[i], (uint64_t)(0xF), __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void STRIDEN_ADD(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes,
                 uint64_t stride){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    thread_args[i].stride = stride;
    pthread_create(&threads[i], &def_attrs, thread_STRIDEN_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads STRIDEN_CAS helper function
void *thread_STRIDEN_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;
  uint64_t stride = arg_ptr->stride;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i += stride){
    __atomic_compare_exchange_n(&ARRAY[i], &ARRAY[i], ARRAY[i], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void STRIDEN_CAS(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes,
                 uint64_t stride){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    thread_args[i].stride = stride;
    pthread_create(&threads[i], &def_attrs, thread_STRIDEN_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads PTRCHASE_ADD helper function
void *thread_PTRCHASE_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    start = __atomic_fetch_add(&IDX[start], (uint64_t)(0x00ull), __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void PTRCHASE_ADD(uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_PTRCHASE_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads PTRCHASE_CAS helper function
void *thread_PTRCHASE_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&IDX[start], &start, IDX[start], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void PTRCHASE_CAS(uint64_t *restrict ARRAY,
                  uint64_t *restrict IDX,
                  uint64_t iters,
                  uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_PTRCHASE_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads SG_ADD helper function
void *thread_SG_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, src, dest, val;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    src  = __atomic_fetch_add(&IDX[i], (uint64_t)(0x00ull), __ATOMIC_RELAXED);
    dest = __atomic_fetch_add(&IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED);
    val = __atomic_fetch_add(&ARRAY[src], (uint64_t)(0x01ull), __ATOMIC_RELAXED);
    __atomic_fetch_add(&ARRAY[dest], val, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void SG_ADD(uint64_t *restrict ARRAY,
            uint64_t *restrict IDX,
            uint64_t iters,
            uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_SG_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads SG_CAS helper function
void *thread_SG_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, src, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	src   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&IDX[i], &src, IDX[i], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&IDX[i+1], &dest, IDX[i+1], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[src], &val, ARRAY[src], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[dest], &ARRAY[dest], val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void SG_CAS(uint64_t *restrict ARRAY,
            uint64_t *restrict IDX,
            uint64_t iters,
            uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_SG_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads CENTRAL_ADD helper function
void *thread_CENTRAL_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  for(i = 0; i < iters; i++){
    __atomic_fetch_add(&ARRAY[0], (uint64_t)(0x1), __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void CENTRAL_ADD(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_CENTRAL_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads CENTRAL_CAS helper function
void *thread_CENTRAL_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i;
  for(i = 0; i < iters; i++){
    __atomic_compare_exchange_n(&ARRAY[0], &ARRAY[0], ARRAY[0], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void CENTRAL_CAS(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_CENTRAL_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads SCATTER_ADD helper function
void *thread_SCATTER_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    dest = __atomic_fetch_add(&IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED);
    val = __atomic_fetch_add(&ARRAY[i], (uint64_t)(0x01ull), __ATOMIC_RELAXED);
    __atomic_fetch_add(&ARRAY[dest], val, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void SCATTER_ADD(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_SCATTER_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads SCATTER_CAS helper function
void *thread_SCATTER_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&IDX[i+1], &dest, IDX[i+1], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[i], &val, ARRAY[i], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[dest], &ARRAY[dest], val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void SCATTER_CAS(uint64_t *restrict ARRAY,
                 uint64_t *restrict IDX,
                 uint64_t iters,
                 uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_SCATTER_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads GATHER_ADD helper function
void *thread_GATHER_ADD(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    dest = __atomic_fetch_add(&IDX[i+1], (uint64_t)(0x00ull), __ATOMIC_RELAXED);
    val = __atomic_fetch_add(&ARRAY[dest], (uint64_t)(0x01ull), __ATOMIC_RELAXED);
    __atomic_fetch_add(&ARRAY[i], val, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void GATHER_ADD(uint64_t *restrict ARRAY,
                uint64_t *restrict IDX,
                uint64_t iters,
                uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_GATHER_ADD, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

// Pthreads GATHER_CAS helper function
void *thread_GATHER_CAS(void *thread_args){

  // Readability/compatible types
  arg_struct* arg_ptr = (arg_struct*) thread_args;
  uint64_t thread_id = arg_ptr->thread_id;
  uint64_t* ARRAY = arg_ptr->ARRAY;
  uint64_t* IDX = arg_ptr->IDX;
  uint64_t iters = arg_ptr->iters;

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    __atomic_compare_exchange_n(&IDX[i+1], &dest, IDX[i+1], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[dest], &val, ARRAY[dest], 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    __atomic_compare_exchange_n(&ARRAY[i], &ARRAY[i], val, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }

  // Worker thread finished
  pthread_exit(0);
}

void GATHER_CAS(uint64_t *restrict ARRAY,
                uint64_t *restrict IDX,
                uint64_t iters,
                uint64_t pes){

  // Thread handles and arg structs
  pthread_t threads[pes];
  arg_struct thread_args[pes];

  // Using default thread attributes
  pthread_attr_t def_attrs;
  pthread_attr_init(&def_attrs);

  // Spawn worker threads
  uint64_t i;
  for(i = 0; i < pes; i++){
    thread_args[i].thread_id = i;
    thread_args[i].ARRAY = ARRAY;
    thread_args[i].IDX = IDX;
    thread_args[i].iters = iters;
    pthread_create(&threads[i], &def_attrs, thread_GATHER_CAS, (void*) &thread_args[i]);
  }

  // Wait for worker thread completion
  for(i = 0; i < pes; i++){
    pthread_join(threads[i], NULL);
  }
}

/* EOF */
