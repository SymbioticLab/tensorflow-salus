/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "devicefactories.h"

#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#endif

namespace tensorflow {

WrappedDeviceSettings::AllocatorFactory WrappedDeviceSettings::m_allocatorFactory = nullptr;

Allocator* WrappedDeviceSettings::getWrapped(Allocator* alloc)
{
    static std::map<Allocator*, std::unique_ptr<Allocator>> cache;

    if (!m_allocatorFactory) {
        return alloc;
    }

    auto &wrapped = cache[alloc];
    if (!wrapped) {
        wrapped = m_allocatorFactory(alloc);
    }
    return wrapped.get();
}

void WrappedDeviceSettings::setWrapperFactory(AllocatorFactory fact)
{
    m_allocatorFactory = fact;
}

// Normal CPU device

class WrappedCPUDevice : public ThreadPoolDevice {
 public:
  WrappedCPUDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, const DeviceLocality& locality,
                   Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator) {}

  ~WrappedCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
      return WrappedDeviceSettings::getWrapped(ThreadPoolDevice::GetAllocator(attr));
  }
};

class WrappedThreadPoolDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    // TODO(zhifengc/tucker): Figure out the number of available CPUs
    // and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/cpu:", i);
      devices->push_back(new WrappedCPUDevice(
          options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("CPU", WrappedThreadPoolDeviceFactory, 888);

#if GOOGLE_CUDA

class WrappedGPUDevice : public BaseGPUDevice
{
public:
    WrappedGPUDevice(const SessionOptions& options, const string& name,
                     Bytes memory_limit, const DeviceLocality& locality, int gpu_id,
                     const string& physical_device_desc, Allocator* gpu_allocator,
                     Allocator* cpu_allocator)
    : BaseGPUDevice(options, name, memory_limit, locality, gpu_id,
                    physical_device_desc, gpu_allocator, cpu_allocator,
                    false /* sync every op */, 1 /* max_streams */)
    {}

    Allocator* GetAllocator(AllocatorAttributes attr) override {
        auto alloc = gpu_allocator_;
        if (attr.on_host()) {
            ProcessState* ps = ProcessState::singleton();
            if (attr.gpu_compatible()) {
                alloc = ps->GetCUDAHostAllocator(0);
            } else {
                alloc = cpu_allocator_;
            }
        }

        alloc = WrappedDeviceSettings::getWrapped(alloc);
        return alloc;
    }
};

class WrappedGPUDeviceFactory : public BaseGPUDeviceFactory
{
private:
    BaseGPUDevice* CreateGPUDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality, int gpu_id,
                                   const string& physical_device_desc,
                                   Allocator* gpu_allocator,
                                   Allocator* cpu_allocator) override
    {
        return new WrappedGPUDevice(options, name, memory_limit, locality, gpu_id,
                                    physical_device_desc, gpu_allocator, cpu_allocator);
    }
};

REGISTER_LOCAL_DEVICE_FACTORY("GPU", WrappedGPUDeviceFactory, 999);

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class WrappedGPUCompatibleCPUDevice : public ThreadPoolDevice {
 public:
  WrappedGPUCompatibleCPUDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, const DeviceLocality& locality,
                   Allocator* allocator)
      : ThreadPoolDevice(options, name, memory_limit, locality, allocator) {}

  ~WrappedGPUCompatibleCPUDevice() override {}

  Allocator* GetAllocator(AllocatorAttributes attr) override {
      if (attr.gpu_compatible()) {
        ProcessState* ps = ProcessState::singleton();
      return WrappedDeviceSettings::getWrapped(ps->GetCUDAHostAllocator(0));
    }
      // Call the parent's implementation.
      return WrappedDeviceSettings::getWrapped(ThreadPoolDevice::GetAllocator(attr));
  }
};

class WrappedGPUCompatibleCPUDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/cpu:", i);
      devices->push_back(new WrappedGPUCompatibleCPUDevice(
          options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
    }

    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("CPU", WrappedGPUCompatibleCPUDeviceFactory, 999);

#endif

} // namespace tensorflow
