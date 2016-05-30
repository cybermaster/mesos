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

#include <stdint.h>

#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <process/collect.hpp>
#include <process/defer.hpp>
#include <process/future.hpp>
#include <process/shared.hpp>

#include <stout/error.hpp>
#include <stout/foreach.hpp>
#include <stout/hashmap.hpp>
#include <stout/option.hpp>
#include <stout/os.hpp>
#include <stout/try.hpp>

#include "linux/cgroups.hpp"

#include "slave/flags.hpp"

#include "slave/containerizer/containerizer.hpp"

#include "slave/containerizer/mesos/isolator.hpp"

#include "slave/containerizer/mesos/isolators/gpu/isolator.hpp"
#include "slave/containerizer/mesos/isolators/gpu/nvml.hpp"

using cgroups::devices::Entry;

using mesos::slave::ContainerConfig;
using mesos::slave::ContainerLaunchInfo;
using mesos::slave::ContainerLimitation;
using mesos::slave::ContainerState;
using mesos::slave::Isolator;

using mesos::internal::slave::Containerizer;

using process::Failure;
using process::Future;
using process::PID;
using process::Shared;

using std::list;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace mesos {
namespace internal {
namespace slave {

// TODO(klueska): Expand this when we support other GPU types.
static constexpr dev_t NVIDIA_MAJOR_DEVICE = 195;

// We need to grant/revoke access to both `/dev/nvidiactl` and
// `/dev/nvidia-uvm` when we grant/revoke access to GPUs in a
// container. Unfortunately, we can't make these const because we need
// to calculate the major/minor device numbers for these entries in
// ::create() using os::stat::rdev().
static cgroups::devices::Entry* NVIDIA_CTL_DEVICE_ENTRY =
  new cgroups::devices::Entry();
static cgroups::devices::Entry* NVIDIA_UVM_DEVICE_ENTRY =
  new cgroups::devices::Entry();

// The default list of devices to whitelist when device isolation is
// turned on. The full list of devices can be found here:
// https://www.kernel.org/doc/Documentation/devices.txt
//
// Device whitelisting is described here:
// https://www.kernel.org/doc/Documentation/cgroup-v1/devices.txt
static const char* DEFAULT_WHITELIST_ENTRIES[] = {
  "c *:* m",      // Make new character devices.
  "b *:* m",      // Make new block devices.
  "c 5:1 rwm",    // /dev/console
  "c 4:0 rwm",    // /dev/tty0
  "c 4:1 rwm",    // /dev/tty1
  "c 136:* rwm",  // /dev/pts/*
  "c 5:2 rwm",    // /dev/ptmx
  "c 10:200 rwm", // /dev/net/tun
  "c 1:3 rwm",    // /dev/null
  "c 1:5 rwm",    // /dev/zero
  "c 1:7 rwm",    // /dev/full
  "c 5:0 rwm",    // /dev/tty
  "c 1:9 rwm",    // /dev/urandom
  "c 1:8 rwm",    // /dev/random
};


NvidiaGpuIsolatorProcess::NvidiaGpuIsolatorProcess(
    const Flags& _flags,
    const string& _hierarchy,
    const Shared<NvidiaGpuAllocator>& _allocator)
  : flags(_flags),
    hierarchy(_hierarchy),
    allocator(_allocator) {}


Try<Isolator*> NvidiaGpuIsolatorProcess::create(
    const Flags& flags,
    const Option<NvidiaComponents>& components)
{
  if (components.isNone()) {
    return Error("Cannot create an `NvidiaGpuIsolatorProcess`"
                 " when `Option<NvidiaComponents>` is None()");
  }

  // Populate the device entries for
  // `/dev/nvidiactl` and `/dev/nvidia-uvm`.
  Try<dev_t> device = os::stat::rdev("/dev/nvidiactl");
  if (device.isError()) {
    return Error(device.error());
  }
  NVIDIA_CTL_DEVICE_ENTRY->selector.type = Entry::Selector::Type::CHARACTER;
  NVIDIA_CTL_DEVICE_ENTRY->selector.major = major(device.get());
  NVIDIA_CTL_DEVICE_ENTRY->selector.minor = minor(device.get());
  NVIDIA_CTL_DEVICE_ENTRY->access.read = true;
  NVIDIA_CTL_DEVICE_ENTRY->access.write = true;
  NVIDIA_CTL_DEVICE_ENTRY->access.mknod = true;

  device = os::stat::rdev("/dev/nvidia-uvm");
  if (device.isError()) {
    return Error(device.error());
  }
  NVIDIA_UVM_DEVICE_ENTRY->selector.type = Entry::Selector::Type::CHARACTER;
  NVIDIA_UVM_DEVICE_ENTRY->selector.major = major(device.get());
  NVIDIA_UVM_DEVICE_ENTRY->selector.minor = minor(device.get());
  NVIDIA_UVM_DEVICE_ENTRY->access.read = true;
  NVIDIA_UVM_DEVICE_ENTRY->access.write = true;
  NVIDIA_UVM_DEVICE_ENTRY->access.mknod = true;

  // Prepare the cgroups device hierarchy.
  //
  // TODO(klueska): Once we have other devices, we will
  // have to do this initialization at a higher level.
  Try<string> hierarchy = cgroups::prepare(
        flags.cgroups_hierarchy,
        "devices",
        flags.cgroups_root);

  if (hierarchy.isError()) {
    return Error(
        "Failed to prepare hierarchy for 'devices' subsystem: " +
        hierarchy.error());
  }

  // Ensure that no other subsystem is attached to the hierarchy.
  //
  // TODO(klueska): Once we have other devices, we will
  // have to do this check at a higher level.
  Try<set<string>> subsystems = cgroups::subsystems(hierarchy.get());
  if (subsystems.isError()) {
    return Error(
        "Failed to get subsystems attached to hierarchy"
        " '" + stringify(hierarchy.get()) + "': " +
        subsystems.error());
  } else if (subsystems.get().size() != 1) {
    return Error(
        "Unexpected subsystems attached to hierarchy"
        " '" + stringify(hierarchy.get()) + "': " +
        stringify(subsystems.get()));
  }

  process::Owned<MesosIsolatorProcess> process(
      new NvidiaGpuIsolatorProcess(
          flags,
          hierarchy.get(),
          components->allocator));

  return new MesosIsolator(process);
}


Future<Nothing> NvidiaGpuIsolatorProcess::recover(
    const list<ContainerState>& states,
    const hashset<ContainerID>& orphans)
{
  list<Future<Nothing>> futures;

  foreach (const ContainerState& state, states) {
    const ContainerID& containerId = state.container_id();
    const string cgroup = path::join(flags.cgroups_root, containerId.value());

    Try<bool> exists = cgroups::exists(hierarchy, cgroup);
    if (exists.isError()) {
      foreachvalue (Info* info, infos) {
        delete info;
      }
      infos.clear();
      return Failure("Failed to check cgroup for container '" +
                     stringify(containerId) + "'");
    }

    if (!exists.get()) {
      VLOG(1) << "Couldn't find cgroup for container " << containerId;
      // This may occur if the executor has exited and the isolator
      // has destroyed the cgroup but the slave dies before noticing
      // this. This will be detected when the containerizer tries to
      // monitor the executor's pid.
      continue;
    }

    infos[containerId] = new Info(containerId, cgroup);

    // Determine which GPUs are allocated to this container.
    Try<vector<cgroups::devices::Entry>> entries =
      cgroups::devices::list(hierarchy, cgroup);

    if (entries.isError()) {
      return Failure("Failed to obtain devices list for cgroup"
                     " '" + cgroup + "': " + entries.error());
    }

    const list<Gpu> allGpus = allocator->allGpus();

    list<Gpu> containerGpus;
    foreach (const cgroups::devices::Entry& entry, entries.get()) {
      for (auto gpu = allGpus.begin(); gpu != allGpus.end(); ++gpu) {
        if (entry.selector.major == gpu->major &&
            entry.selector.minor == gpu->minor) {
          containerGpus.push_back(*gpu);
          break;
        }
      }
    }

    futures.push_back(allocator->allocate(containerGpus)
      .then([=] () -> Future<Nothing> {
        infos[containerId]->allocated = containerGpus;
        return Nothing();
      }));
  }

  return collect(futures)
    .then([=] () -> Future<Nothing> {
      // Remove orphan cgroups.
      Try<vector<string>> cgroups = cgroups::get(hierarchy, flags.cgroups_root);
      if (cgroups.isError()) {
        foreachvalue (Info* info, infos) {
          delete info;
        }
        infos.clear();
        return Failure(cgroups.error());
      }

      foreach (const string& cgroup, cgroups.get()) {
        // Ignore the slave cgroup (see the --agent_subsystems flag).
        // TODO(idownes): Remove this when the cgroups layout is updated,
        // see MESOS-1185.
        if (cgroup == path::join(flags.cgroups_root, "slave")) {
          continue;
        }

        ContainerID containerId;
        containerId.set_value(Path(cgroup).basename());

        if (infos.contains(containerId)) {
          continue;
        }

        // Known orphan cgroups will be destroyed by the containerizer
        // using the normal cleanup path. See MESOS-2367 for details.
        if (orphans.contains(containerId)) {
          infos[containerId] = new Info(containerId, cgroup);
          continue;
        }

        LOG(INFO) << "Removing unknown orphaned cgroup '" << cgroup << "'";

        // We don't wait on the destroy as we don't want to block recovery.
        cgroups::destroy(hierarchy, cgroup, cgroups::DESTROY_TIMEOUT);
      }

      return Nothing();
    });
}


Future<Option<ContainerLaunchInfo>> NvidiaGpuIsolatorProcess::prepare(
    const ContainerID& containerId,
    const mesos::slave::ContainerConfig& containerConfig)
{
  if (infos.contains(containerId)) {
    return Failure("Container has already been prepared");
  }

  // TODO(bmahler): Don't insert into 'infos' unless we create the
  // cgroup successfully. It's safe for now because 'cleanup' gets
  // called if we return a Failure, but cleanup will fail because the
  // cgroup does not exist when cgroups::destroy is called.
  Info* info = new Info(
      containerId, path::join(flags.cgroups_root, containerId.value()));

  infos[containerId] = info;

  Try<bool> exists = cgroups::exists(hierarchy, info->cgroup);
  if (exists.isError()) {
    return Failure("Failed to prepare isolator: " + exists.error());
  } else if (exists.get()) {
    return Failure("Failed to prepare isolator: cgroup already exists");
  }

  Try<Nothing> create = cgroups::create(hierarchy, info->cgroup);
  if (create.isError()) {
    return Failure("Failed to prepare isolator: " + create.error());
  }

  // Chown the cgroup so the executor can create nested cgroups. Do
  // not recurse so the control files are still owned by the slave
  // user and thus cannot be changed by the executor.
  if (containerConfig.has_user()) {
    Try<Nothing> chown = os::chown(
        containerConfig.user(),
        path::join(hierarchy, info->cgroup),
        false);
    if (chown.isError()) {
      return Failure("Failed to prepare isolator: " + chown.error());
    }
  }

  // When a devices cgroup is first created, its whitelist inherits
  // all devices from its parent's whitelist (i.e., "a *:* rwm" by
  // default). In theory, we should be able to add and remove devices
  // from the whitelist by writing to the respective `devices.allow`
  // and `devices.deny` files associated with the cgroup. However, the
  // semantics of the whitelist are such that writing to the deny file
  // will only remove entries in the whitelist that are explicitly
  // listed in there (i.e., denying "b 1:3 rwm" when the whitelist
  // only contains "a *:* rwm" will not modify the whitelist because
  // "b 1:3 rwm" is not explicitly listed). Although the whitelist
  // doesn't change, access to the device is still denied as expected
  // (there is just no way of querying the system to detect it).
  // Because of this, we first deny access to all devices and
  // selectively add some back in so we can control the entries in the
  // whitelist explicitly.
  //
  // TODO(klueska): This should really be part of a generic device
  // isolator rather than stuck in the Nvidia GPU specific device
  // isolator. We should move this once we have such a component.
  cgroups::devices::Entry all;
  all.selector.type = Entry::Selector::Type::ALL;
  all.selector.major = None();
  all.selector.minor = None();
  all.access.read = true;
  all.access.write = true;
  all.access.mknod = true;

  Try<Nothing> deny = cgroups::devices::deny(hierarchy, info->cgroup, all);

  if (deny.isError()) {
    return Failure("Failed to deny all devices: " + deny.error());
  }

  foreach (const char* _entry, DEFAULT_WHITELIST_ENTRIES) {
    Try<cgroups::devices::Entry> entry =
      cgroups::devices::Entry::parse(_entry);

    CHECK_SOME(entry);

    Try<Nothing> allow =
      cgroups::devices::allow(hierarchy, info->cgroup, entry.get());

    if (allow.isError()) {
      return Failure("Failed to whitelist default device"
                     " '" + stringify(entry.get()) + "': " + allow.error());
    }
  }

  // Grant access to /dev/nvidiactl and /dev/nvida-uvm
  map<string, const cgroups::devices::Entry*> entries = {
    { "/dev/nvidiactl", NVIDIA_CTL_DEVICE_ENTRY },
    { "/dev/nvidia-uvm", NVIDIA_UVM_DEVICE_ENTRY },
  };

  foreachkey (const string& device, entries) {
    Try<Nothing> allow = cgroups::devices::allow(
        hierarchy, info->cgroup, *entries[device]);

    if (allow.isError()) {
      return Failure("Failed to grant cgroups access to"
                     " '" + device + "': " + allow.error());
    }
  }

  return update(containerId, containerConfig.executor_info().resources())
    .then([]() -> Future<Option<ContainerLaunchInfo>> {
      return None();
    });
}


Future<Nothing> NvidiaGpuIsolatorProcess::isolate(
    const ContainerID& containerId,
    pid_t pid)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  Try<Nothing> assign = cgroups::assign(
      hierarchy, info->cgroup, pid);

  if (assign.isError()) {
    LOG(ERROR) << "Failed to assign container '" << info->containerId << "'"
               << " to cgroup '" << path::join(hierarchy, info->cgroup) << "': "
               << assign.error();

    return Failure("Failed to isolate container: " + assign.error());
  }

  return Nothing();
}


Future<Nothing> NvidiaGpuIsolatorProcess::update(
    const ContainerID& containerId,
    const Resources& resources)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  Option<double> gpus = resources.gpus();

  // Make sure that the `gpus` resource is not fractional.
  // We rely on scalar resources only having 3 digits of precision.
  if (static_cast<long long>(gpus.getOrElse(0.0) * 1000.0) % 1000 != 0) {
    return Failure("The 'gpus' resource must be an unsigned integer");
  }

  size_t requested = static_cast<size_t>(resources.gpus().getOrElse(0.0));

  // Update the GPU allocation to reflect the new total.
  if (requested > info->allocated.size()) {
    size_t additional = requested - info->allocated.size();

    return allocator->allocate(additional)
      .then([=] (Option<list<Gpu>> allocated) -> Future<Nothing> {
        if (!allocated.isSome()) {
          return Failure("Not enough GPUs available to reserve"
                         " " + stringify(additional) + " additional GPUs");
        }

        foreach (const Gpu& gpu, allocated.get()) {
          cgroups::devices::Entry entry;
          entry.selector.type = Entry::Selector::Type::CHARACTER;
          entry.selector.major = gpu.major;
          entry.selector.minor = gpu.minor;
          entry.access.read = true;
          entry.access.write = true;
          entry.access.mknod = true;

          Try<Nothing> allow = cgroups::devices::allow(
              hierarchy, info->cgroup, entry);

          if (allow.isError()) {
            return Failure("Failed to grant cgroups access to GPU device"
                           " '" + stringify(entry) + "': " + allow.error());
          }
        }

        info->allocated = allocated.get();

        return Nothing();
      });
  } else if (requested < info->allocated.size()) {
    size_t fewer = info->allocated.size() - requested;

    list<Gpu> deallocated;

    for (size_t i = 0; i < fewer; i++) {
      const Gpu& gpu = info->allocated.front();

      cgroups::devices::Entry entry;
      entry.selector.type = Entry::Selector::Type::CHARACTER;
      entry.selector.major = gpu.major;
      entry.selector.minor = gpu.minor;
      entry.access.read = true;
      entry.access.write = true;
      entry.access.mknod = true;

      Try<Nothing> deny = cgroups::devices::deny(
          hierarchy, info->cgroup, entry);

      if (deny.isError()) {
        return Failure("Failed to deny cgroups access to GPU device"
                       " '" + stringify(entry) + "': " + deny.error());
      }

      deallocated.push_back(gpu);
      info->allocated.pop_front();
    }

    return allocator->deallocate(deallocated);
  }

  return Nothing();
}


Future<ResourceStatistics> NvidiaGpuIsolatorProcess::usage(
    const ContainerID& containerId)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  // TODO(rtodd): Obtain usage information from NVML.

  ResourceStatistics result;
  return result;
}


Future<Nothing> NvidiaGpuIsolatorProcess::cleanup(
    const ContainerID& containerId)
{
  // Multiple calls may occur during test clean up.
  if (!infos.contains(containerId)) {
    VLOG(1) << "Ignoring cleanup request for unknown container " << containerId;

    return Nothing();
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  // TODO(klueska): We will need to move this up a level and come up
  // with a different strategy for cleaning up the cgroups for gpus
  // once we have a generic device isolator.
  return cgroups::destroy(hierarchy, info->cgroup, cgroups::DESTROY_TIMEOUT)
    .onAny(defer(PID<NvidiaGpuIsolatorProcess>(this),
                 &NvidiaGpuIsolatorProcess::_cleanup,
                 containerId,
                 lambda::_1));
}


Future<Nothing> NvidiaGpuIsolatorProcess::_cleanup(
    const ContainerID& containerId,
    const Future<Nothing>& future)
{
  if (!infos.contains(containerId)) {
    return Failure("Unknown container");
  }

  Info* info = CHECK_NOTNULL(infos[containerId]);

  Future<Nothing> deallocate = allocator->deallocate(info->allocated);

  return collect(future, deallocate)
    .onFailed([=] (const string& message) -> Future<Nothing> {
      return Failure(
          "Failed to clean up container " +
          stringify(containerId) + ": " + message);
    })
    .then([=] () -> Future<Nothing> {
      delete info;
      infos.erase(containerId);

      return Nothing();
    });
}

} // namespace slave {
} // namespace internal {
} // namespace mesos {
