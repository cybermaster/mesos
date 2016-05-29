// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NVIDIA_GPU_ALLOCATOR_HPP__
#define __NVIDIA_GPU_ALLOCATOR_HPP__

#include <list>

#include <mesos/resources.hpp>

#include <process/future.hpp>
#include <process/process.hpp>

#include <stout/option.hpp>
#include <stout/try.hpp>

#include "slave/flags.hpp"

namespace mesos {
namespace internal {
namespace slave {

struct Gpu
{
  unsigned int major;
  unsigned int minor;

  bool operator==(const Gpu& that) const
  {
    return major == that.major && minor == that.minor;
  }

  bool operator!=(const Gpu& that) const
  {
    return major != that.major || minor != that.minor;
  }
};


// Forward declaration.
class NvidiaGpuAllocatorProcess;


class NvidiaGpuAllocator
{
public:
  static Try<Resources> resources(const Flags& flags);

  static Try<NvidiaGpuAllocator*> create(const Flags& flags);
  ~NvidiaGpuAllocator();

  const std::list<Gpu>& allGpus() const;

  process::Future<Option<Gpu>> allocate() const;
  process::Future<Option<std::list<Gpu>>> allocate(unsigned int count) const;
  process::Future<Nothing> allocate(const Gpu& gpu) const;
  process::Future<Nothing> allocate(const std::list<Gpu>& gpus) const;

  process::Future<Nothing> deallocate(const Gpu& gpu) const;
  process::Future<Nothing> deallocate(const std::list<Gpu>& gpus) const;

private:
  NvidiaGpuAllocator(const std::list<Gpu>& gpus);
  NvidiaGpuAllocator(const NvidiaGpuAllocator&) = delete;
  void operator=(const NvidiaGpuAllocator&) = delete;

  std::list<Gpu> gpus;
  process::Owned<NvidiaGpuAllocatorProcess> process;
};


class NvidiaGpuAllocatorProcess
  : public process::Process<NvidiaGpuAllocatorProcess>
{
public:
  NvidiaGpuAllocatorProcess(const std::list<Gpu>& gpus);
  ~NvidiaGpuAllocatorProcess() {}

  process::Future<Option<Gpu>> allocate();
  process::Future<Option<std::list<Gpu>>> allocateCount(unsigned int count);
  process::Future<Nothing> allocateSpecific(const Gpu& gpu);
  process::Future<Nothing> allocateList(const std::list<Gpu>& gpus);

  process::Future<Nothing> deallocate(const Gpu& gpu);
  process::Future<Nothing> deallocateList(const std::list<Gpu>& gpu);

private:
  std::list<Gpu> available;
  std::list<Gpu> taken;
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __NVIDIA_GPU_ALLOCATOR_HPP__
