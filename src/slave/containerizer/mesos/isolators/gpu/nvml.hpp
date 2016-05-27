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

#ifndef __GPU_NVIDIA_MANAGEMENT_LIBRARY_HPP__
#define __GPU_NVIDIA_MANAGEMENT_LIBRARY_HPP__

#include <nvidia/gdk/nvml.h>

#include <stout/try.hpp>

namespace mesos {
namespace internal {
namespace slave {

class NvidiaManagementLibrary
{
public:
  static bool isAvailable();
  static Try<Nothing> initialize();
  static const NvidiaManagementLibrary& nvml();

  Try<unsigned int> deviceGetCount() const;
  Try<nvmlDevice_t> deviceGetHandleByIndex(unsigned int index) const;
  Try<unsigned int> deviceGetMinorNumber(nvmlDevice_t handle) const;

private:
  NvidiaManagementLibrary(
      nvmlReturn_t (*_nvmlDeviceGetCount)(unsigned int*),
      nvmlReturn_t (*_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*),
      nvmlReturn_t (*_nvmlDeviceGetMinorNumber)(nvmlDevice_t, unsigned int*),
      const char* (*_nvmlErrorString)(nvmlReturn_t))
    : nvmlDeviceGetCount(_nvmlDeviceGetCount),
      nvmlDeviceGetHandleByIndex(_nvmlDeviceGetHandleByIndex),
      nvmlDeviceGetMinorNumber(_nvmlDeviceGetMinorNumber),
      nvmlErrorString(_nvmlErrorString) {}

  ~NvidiaManagementLibrary() {}
  NvidiaManagementLibrary(const NvidiaManagementLibrary&) = delete;
  void operator=(const NvidiaManagementLibrary&) = delete;

  static const NvidiaManagementLibrary* instance;

  nvmlReturn_t (*nvmlDeviceGetCount)(unsigned int*);
  nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
  nvmlReturn_t (*nvmlDeviceGetMinorNumber)(nvmlDevice_t, unsigned int*);
  const char* (*nvmlErrorString)(nvmlReturn_t);
};

} // namespace slave {
} // namespace internal {
} // namespace mesos {

#endif // __GPU_NVIDIA_ISOLATOR_NVML_HPP__
