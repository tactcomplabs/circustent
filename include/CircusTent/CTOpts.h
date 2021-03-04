//
// _CTOpts_h_
//
// Copyright (C) 2017-2021 Tactical Computing Laboratories, LLC
// All Rights Reserved
// contact@tactcomplabs.com
//
// See LICENSE in the top level directory for licensing details
//

#ifndef _CTOPTS_H_
#define _CTOPTS_H_

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "CircusTent/CTBaseImpl.h"

#define CT_VERSION_MAJOR 0
#define CT_VERSION_MINOR 1

/** BenchType struct: defines an individual benchmark table entry */
typedef struct{
  const std::string Name;         ///< BenchType: Benchmark name
  std::string Arg;                ///< BenchType: Benchmark arguments
  const std::string Notes;        ///< BenchType: Notes for the list printer
  CTBaseImpl::CTBenchType BType;  ///< BenchType: Type of benchmark
  CTBaseImpl::CTAtomType AType;   ///< BenchType: Atomic type of the benchmark
  bool Enabled;                   ///< BenchType: Is the benchmark enabled
  bool ReqArg;                    ///< BenchType: Is there a required argument
}BenchType;

class CTOpts{
private:
  bool isHelp;                ///< determines if the help option has been selected
  bool isList;                ///< list the benchmarks

  uint64_t memSize;           ///< size of the memory array in bytes
  uint64_t iters;             ///< number of iterations per thread
  uint64_t pes;               ///< number of pe's
  uint64_t stride;            ///< stride between accesses
  int l_argc;                 ///< main argc
  char **l_argv;              ///< main argv

  /// Prints the help menu
  void PrintHelp();

  /// Prints the benchmark list
  void PrintBench();

  /// Checks and enables the benchmark entry
  bool EnableBench( std::string Bench );

public:

  /// default constructor
  CTOpts();

  /// default destructor
  ~CTOpts();

  /// Returns whether the help option was selected
  bool IsHelp() { return isHelp; }

  /// Returns whether the list options is selected
  bool IsList() { return isList; }

  /// Parse the input args
  bool ParseOpts(int argc, char **argv);

  /// Retrieve the benchmark type
  CTBaseImpl::CTBenchType GetBenchType();

  /// Retrieve the atomic type
  CTBaseImpl::CTAtomType GetAtomType();

  /// Retrieve the memory size
  uint64_t GetMemSize() { return memSize; }

  /// Retrieve the number of iterations
  uint64_t GetIters() { return iters; }

  /// Retrieve the number of PEs
  uint64_t GetPEs() { return pes; }

  /// Retrieve the stride in elems
  uint64_t GetStride() { return stride; }

  /// Retrieve the argc value
  int GetArgc() { return l_argc; }

  /// Retrieve the argv value
  char **GetArgv() { return l_argv; }
};

#endif

// EOF
