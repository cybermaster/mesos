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

#include <nvidia/gdk/nvml.h>

#include <string>

#include <process/once.hpp>

#include <stout/exit.hpp>
#include <stout/nothing.hpp>
#include <stout/stringify.hpp>
#include <stout/try.hpp>

#include <stout/posix/dynamiclibrary.hpp>

#include "slave/containerizer/mesos/isolators/gpu/nvml.hpp"

using process::Once;

using std::string;

namespace mesos {
namespace internal {
namespace slave {

const NvidiaManagementLibrary* NvidiaManagementLibrary::instance = NULL;


Try<Nothing> NvidiaManagementLibrary::initialize()
{
  static Once* initialized = new Once();
  static Option<Error>* error = new Option<Error>();

  if (initialized->once()) {
    if (error->isSome()) {
      return error->get();
    }
    return Nothing();
  }

  static DynamicLibrary* library = new DynamicLibrary();

  Try<Nothing> open = library->open("libnvidia-ml.so");
  if (open.isError()) {
    *error = open.error();
    initialized->done();
    return error->get();
  }

  Try<void*> symbol = library->loadSymbol("nvmlInit");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlInit = (nvmlReturn_t (*)())symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetCount");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetCount = (nvmlReturn_t (*)(unsigned int*))symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetHandleByIndex");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetHandleByIndex =
    (nvmlReturn_t (*)(unsigned int, nvmlDevice_t*))symbol.get();

  symbol = library->loadSymbol("nvmlDeviceGetMinorNumber");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlDeviceGetMinorNumber =
    (nvmlReturn_t (*)(nvmlDevice_t, unsigned int*))symbol.get();

  symbol = library->loadSymbol("nvmlErrorString");
  if (symbol.isError()) {
    *error = symbol.error();
    initialized->done();
    return error->get();
  }
  auto nvmlErrorString = (const char* (*)(nvmlReturn_t))symbol.get();

  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    *error = Error("nvmlInit failed: " +  stringify(nvmlErrorString(result)));
    initialized->done();
    return error->get();
  }

  instance = new NvidiaManagementLibrary(
      nvmlDeviceGetCount,
      nvmlDeviceGetHandleByIndex,
      nvmlDeviceGetMinorNumber,
      nvmlErrorString);

  initialized->done();

  return Nothing();
}


bool NvidiaManagementLibrary::isAvailable()
{
  if (instance == NULL) {
    Try<Nothing> result = initialize();
    if (result.isError()) {
      return false;
    }
  }
  return true;
}


const NvidiaManagementLibrary& NvidiaManagementLibrary::nvml()
{
  return *CHECK_NOTNULL(instance);
}


Try<unsigned int> NvidiaManagementLibrary::deviceGetCount() const
{
  unsigned int count;
  nvmlReturn_t result = nvmlDeviceGetCount(&count);
  if (result != NVML_SUCCESS) {
    return Error("nvmlDeviceGetCount failed: " +
                 stringify(nvmlErrorString(result)));
  }
  return count;
}


Try<nvmlDevice_t> NvidiaManagementLibrary::deviceGetHandleByIndex(
    unsigned int index) const
{
  nvmlDevice_t handle;
  nvmlReturn_t result = nvmlDeviceGetHandleByIndex(index, &handle);
  if (result == NVML_ERROR_INVALID_ARGUMENT) {
    return Error("GPU device " + stringify(handle) + " not found");
  }
  if (result != NVML_SUCCESS) {
    return Error("nvmlDeviceGetHandleByIndex failed: " +
                 stringify(nvmlErrorString(result)));
  }
  return handle;
}


Try<unsigned int> NvidiaManagementLibrary::deviceGetMinorNumber(
    nvmlDevice_t handle) const
{
  unsigned int minor;
  nvmlReturn_t result = nvmlDeviceGetMinorNumber(handle, &minor);
  if (result != NVML_SUCCESS) {
    return Error("nvmlGetMinorNumber failed: " +
                 stringify(nvmlErrorString(result)));
  }
  return minor;
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
