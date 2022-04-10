/*
 * _CT_CPP_STD_IMPL_CPP_
 *
 * Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
 * All Rights Reserved
 * contact@tactcomplabs.com
 *
 * See LICENSE in the top level directory for licensing details
 */

#include "CT_CPP_STD.h"

/* CPP Standard Atomics Benchmark Implementations */

// RAND_ADD kernel function
void CT_CPP_STD::RAND_ADD(uint64_t thread_id,
                          std::atomic<std::uint64_t> *barrier_ctr,
                          double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    Array[Idx[i]].fetch_add((uint64_t)(0x1), std::memory_order_relaxed);
  }
}

// RAND_CAS kernel function
void CT_CPP_STD::RAND_CAS(uint64_t thread_id,
                          std::atomic<std::uint64_t> *barrier_ctr,
                          double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[Idx[start+i]];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  for(i = 0; i < iters; i++){
    Array[Idx[start+i]].compare_exchange_strong(expected[i], Array[Idx[start+i]], std::memory_order_relaxed);
  }
}

// STRIDE1_ADD kernel function
void CT_CPP_STD::STRIDE1_ADD(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    Array[i].fetch_add((uint64_t)(0xF), std::memory_order_relaxed);
  }
}

// STRIDE1_CAS kernel function
void CT_CPP_STD::STRIDE1_CAS(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[start+i];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  for(i = 0; i < iters; i++){
    Array[start+i].compare_exchange_strong(expected[i], Array[start+i], std::memory_order_relaxed);
  }
}

// STRIDEN_ADD kernel function
void CT_CPP_STD::STRIDEN_ADD(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i += stride){
    Array[i].fetch_add((uint64_t)(0xF), std::memory_order_relaxed);
  }
}

// STRIDEN_ADD kernel function
void CT_CPP_STD::STRIDEN_CAS(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[start+(stride*i)];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  for(i = 0; i < iters; i++){
    Array[start+(stride*i)].compare_exchange_strong(expected[i], Array[start+(stride*i)], std::memory_order_relaxed);
  }
}

// PTRCHASE_ADD kernel function
void CT_CPP_STD::PTRCHASE_ADD(uint64_t thread_id,
                              std::atomic<std::uint64_t> *barrier_ctr,
                              double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    start = Idx[start].fetch_add((uint64_t)(0x00ull), std::memory_order_relaxed);
  }
}

// PTRCHASE_CAS kernel function
void CT_CPP_STD::PTRCHASE_CAS(uint64_t thread_id,
                              std::atomic<std::uint64_t> *barrier_ctr,
                              double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    Idx[start].compare_exchange_strong(start, Idx[start], std::memory_order_relaxed);
  }
}

// SG_ADD kernel function
void CT_CPP_STD::SG_ADD(uint64_t thread_id,
                        std::atomic<std::uint64_t> *barrier_ctr,
                        double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i, src, dest, val;
  uint64_t start = thread_id * iters;
  for(i = start; i < (start + iters); i++){
    src = Idx[i].fetch_add((uint64_t)(0x00ull), std::memory_order_relaxed);
    dest = Idx[i+1].fetch_add((uint64_t)(0x00ull), std::memory_order_relaxed);
    val = Array[src].fetch_add((uint64_t)(0x01ull), std::memory_order_relaxed);
    Array[dest].fetch_add(val, std::memory_order_relaxed);
  }
}

// SG_CAS kernel function
void CT_CPP_STD::SG_CAS(uint64_t thread_id,
                        std::atomic<std::uint64_t> *barrier_ctr,
                        double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[Idx[start+i+1]];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t src, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	src   = 0x00ull;
	dest  = 0x00ull;
  for(i = 0; i < iters; i++){
    Idx[start+i].compare_exchange_strong(src, Idx[start+i], std::memory_order_relaxed);
    Idx[start+i+1].compare_exchange_strong(dest, Idx[start+i+1], std::memory_order_relaxed);
    Array[src].compare_exchange_strong(val, Array[src], std::memory_order_relaxed);
    // AMO #4 issue - expected may not equal Array[dest] due to previous ops
    // Result: expected[i] <- Array[dest] rather than Array[dest] <- val
    Array[dest].compare_exchange_strong(expected[i], val, std::memory_order_relaxed);
  }
}

// CENTRAL_ADD kernel function
void CT_CPP_STD::CENTRAL_ADD(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  for(i = 0; i < iters; i++){
    Array[0].fetch_add((uint64_t)(0x1), std::memory_order_relaxed);
  }
}

// CENTRAL_CAS kernel function
void CT_CPP_STD::CENTRAL_CAS(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i;
  uint64_t expected = Array[0];
  for(i = 0; i < iters; i++){
    Array[0].compare_exchange_strong(expected, Array[0], std::memory_order_relaxed);
  }
}

// SCATTER_ADD kernel function
void CT_CPP_STD::SCATTER_ADD(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    dest = Idx[i+1].fetch_add((uint64_t)(0x00ull), std::memory_order_relaxed);
    val = Array[i].fetch_add((uint64_t)(0x01ull), std::memory_order_relaxed);
    Array[dest].fetch_add(val, std::memory_order_relaxed);
  }
}

// SCATTER_CAS kernel function
void CT_CPP_STD::SCATTER_CAS(uint64_t thread_id,
                             std::atomic<std::uint64_t> *barrier_ctr,
                             double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[IDX[start+i+1]];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t dest, val;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = 0; i < iters; i++){
    Idx[start+i+1].compare_exchange_strong(dest, Idx[start+i+1], std::memory_order_relaxed);
    Array[start+i].compare_exchange_strong(val, Array[start+i], std::memory_order_relaxed);
    // AMO #3 issue - expected may not equal Array[dest] due to previous ops
    // Result: expected[i] <- Array[dest] rather than Array[dest] <- val
    Array[dest].compare_exchange_strong(expected[i], val, std::memory_order_relaxed);
  }
}

// GATHER_ADD kernel function
void CT_CPP_STD::GATHER_ADD(uint64_t thread_id,
                            std::atomic<std::uint64_t> *barrier_ctr,
                            double* start_time ){

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t i, dest, val;
  uint64_t start = thread_id * iters;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = start; i < (start + iters); i++){
    dest = Idx[i+1].fetch_add((uint64_t)(0x00ull), std::memory_order_relaxed);
    val = Array[dest].fetch_add((uint64_t)(0x01ull), std::memory_order_relaxed);
    Array[i].fetch_add(val, std::memory_order_relaxed);
  }
}

// GATHER_CAS kernel function
void CT_CPP_STD::GATHER_CAS(uint64_t thread_id,
                            std::atomic<std::uint64_t> *barrier_ctr,
                            double* start_time ){

  // Set up array of expected uint64_t values
  uint64_t i;
  uint64_t expected[iters];
  uint64_t start = thread_id * iters;
  for(i = 0; i < iters; i++){
    expected[i] = Array[start+i];
  }

  // Wait for all threads to be spawned
  MyBarrier(barrier_ctr);

  // Thread 0 write kernel StartTime
  if(thread_id == 0){
    *start_time = MySecond();
  }

  // Perform atomic ops
  uint64_t dest, val;
	val   = 0x00ull;
	dest  = 0x00ull;
  for(i = 0; i < iters; i++){
    Idx[start+i+1].compare_exchange_strong(dest, Idx[start+i+1], std::memory_order_relaxed);
    Array[dest].compare_exchange_strong(val, Array[dest], std::memory_order_relaxed);
    Array[start+i].compare_exchange_strong(expected[i], val, std::memory_order_relaxed);
  }
}

/* EOF */
