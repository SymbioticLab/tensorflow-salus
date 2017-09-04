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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#endif

#include <atomic>

namespace tensorflow {
namespace remote {

WrappedDeviceSettings::AllocatorFactory WrappedDeviceSettings::m_allocatorFactory = nullptr;

Allocator *WrappedDeviceSettings::getWrapped(Allocator *alloc, Device *device)
{
    static std::map<Allocator *, std::unique_ptr<Allocator>> cache;
    static mutex mu;

    if (!m_allocatorFactory) {
        return alloc;
    }

    mutex_lock l(mu);
    auto &wrapped = cache[alloc];
    if (!wrapped) {
        wrapped = m_allocatorFactory(alloc, device);
    }
    return wrapped.get();
}

void WrappedDeviceSettings::setWrapperFactory(AllocatorFactory fact)
{
    m_allocatorFactory = fact;
}

// Normal CPU device

class WrappedThreadPoolDevice : public ThreadPoolDevice
{
public:
    WrappedThreadPoolDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
                            const DeviceLocality &locality, Allocator *allocator)
        : ThreadPoolDevice(options, name, memory_limit, locality, allocator)
    {
    }

    ~WrappedThreadPoolDevice() override
    {
    }

    Allocator *GetAllocator(AllocatorAttributes attr) override
    {
        return WrappedDeviceSettings::getWrapped(ThreadPoolDevice::GetAllocator(attr), this);
    }
};

class WrappedThreadPoolDeviceFactory : public DeviceFactory
{
public:
    Status CreateDevices(const SessionOptions &options, const string &name_prefix,
                         std::vector<Device *> *devices) override
    {
        // TODO(zhifengc/tucker): Figure out the number of available CPUs
        // and/or NUMA configuration.
        int n = 1;
        auto iter = options.config.device_count().find("CPU");
        if (iter != options.config.device_count().end()) {
            n = iter->second;
        }
        for (int i = 0; i < n; i++) {
            string name = strings::StrCat(name_prefix, "/cpu:", i);
            devices->push_back(new WrappedThreadPoolDevice(options, name, Bytes(256 << 20), DeviceLocality(),
                                                           cpu_allocator()));
        }

        return Status::OK();
    }
};

#if GOOGLE_CUDA

class WrappedGPUDevice : public BaseGPUDevice
{
public:
    WrappedGPUDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
                     const DeviceLocality &locality, int gpu_id, const string &physical_device_desc,
                     Allocator *gpu_allocator, Allocator *cpu_allocator)
        : BaseGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc, gpu_allocator,
                        cpu_allocator, false /* sync every op */, 1 /* max_streams */)
        , gpu_id(gpu_id)
    {
    }

    Allocator *GetAllocator(AllocatorAttributes attr) override
    {
        auto alloc = gpu_allocator_;
        if (attr.on_host()) {
            ProcessState *ps = ProcessState::singleton();
            if (attr.gpu_compatible()) {
                alloc = ps->GetCUDAHostAllocator(0);
            } else {
                alloc = cpu_allocator_;
            }
        }

        alloc = WrappedDeviceSettings::getWrapped(alloc, this);
        return alloc;
    }
protected:
    int gpu_id = -1;
};

class WrappedGPUDeviceFactory : public BaseGPUDeviceFactory
{
private:
    BaseGPUDevice *CreateGPUDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
                                   const DeviceLocality &locality, int gpu_id,
                                   const string &physical_device_desc, Allocator *gpu_allocator,
                                   Allocator *cpu_allocator) override
    {
        return new WrappedGPUDevice(options, name, memory_limit, locality, gpu_id, physical_device_desc,
                                    gpu_allocator, cpu_allocator);
    }
};

//------------------------------------------------------------------------------
// A CPUDevice that optimizes for interaction with GPUs in the
// process.
// -----------------------------------------------------------------------------
class WrappedGPUCompatibleCPUDevice : public ThreadPoolDevice
{
public:
    WrappedGPUCompatibleCPUDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
                                  const DeviceLocality &locality, Allocator *allocator)
        : ThreadPoolDevice(options, name, memory_limit, locality, allocator)
    {
    }

    ~WrappedGPUCompatibleCPUDevice() override
    {
    }

    Allocator *GetAllocator(AllocatorAttributes attr) override
    {
        if (attr.gpu_compatible()) {
            ProcessState *ps = ProcessState::singleton();
            return WrappedDeviceSettings::getWrapped(ps->GetCUDAHostAllocator(0), this);
        }
        // Call the parent's implementation.
        return WrappedDeviceSettings::getWrapped(ThreadPoolDevice::GetAllocator(attr), this);
    }
};

class WrappedGPUCompatibleCPUDeviceFactory : public DeviceFactory
{
public:
    Status CreateDevices(const SessionOptions &options, const string &name_prefix,
                         std::vector<Device *> *devices) override
    {
        int n = 1;
        auto iter = options.config.device_count().find("CPU");
        if (iter != options.config.device_count().end()) {
            n = iter->second;
        }
        for (int i = 0; i < n; i++) {
            string name = strings::StrCat(name_prefix, "/cpu:", i);
            devices->push_back(new WrappedGPUCompatibleCPUDevice(options, name, Bytes(256 << 20),
                                                                 DeviceLocality(), cpu_allocator()));
        }

        return Status::OK();
    }
};

#endif // GOOGLE_CUDA

void WrappedDeviceSettings::maybeRegisterWrappedDeviceFactories()
{
    static std::atomic_flag done = ATOMIC_FLAG_INIT;
    if (done.test_and_set()) {
        return;
    }

    REGISTER_LOCAL_DEVICE_FACTORY("CPU", WrappedThreadPoolDeviceFactory, 888);

#if GOOGLE_CUDA
    REGISTER_LOCAL_DEVICE_FACTORY("GPU", WrappedGPUDeviceFactory, 999);
    REGISTER_LOCAL_DEVICE_FACTORY("CPU", WrappedGPUCompatibleCPUDeviceFactory, 999);
#endif // GOOGLE_CUDA
}

} // namespace remote
} // namespace tensorflow
