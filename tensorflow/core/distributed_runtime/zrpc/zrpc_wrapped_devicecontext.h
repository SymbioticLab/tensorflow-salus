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
 */

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_WRAPPEDDEVICECONTEXT_H
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_WRAPPEDDEVICECONTEXT_H

#include "tensorflow/core/common_runtime/device.h"

#include <functional>

namespace tensorflow {

class WrapperDeviceContext : public DeviceContext
{
public:
    using WrapperFunction = std::function<Allocator*(Allocator*)>;

private:
    Device *m_device;
    WrapperFunction m_allocWrapper;

    DeviceContext *m_actualCtx;

public:
    // Takes one ref on `actual`
    WrapperDeviceContext(Device *dev, WrapperFunction allocWrapper, DeviceContext *actual)
        : m_device(dev)
        , m_allocWrapper(allocWrapper)
        , m_actualCtx(actual)
    {
        assert(m_device);
        if (m_actualCtx)
            m_actualCtx->Ref();
    }

    ~WrapperDeviceContext() override
    {
        if (m_actualCtx)
            m_actualCtx->Unref();
    }

    perftools::gputools::Stream *stream() const override
    {
        if (m_actualCtx) {
            return m_actualCtx->stream();
        } else {
            return DeviceContext::stream();
        }
    }

    void MaintainLifetimeOnStream(const Tensor *t,
                                  perftools::gputools::Stream *stream) const override
    {
        if (m_actualCtx) {
            return m_actualCtx->MaintainLifetimeOnStream(t, stream);
        } else {
            return DeviceContext::MaintainLifetimeOnStream(t, stream);
        }
    }

    void CopyCPUTensorToDevice(const Tensor *cpu_tensor, Device *device,
                               Tensor *device_tensor,
                               StatusCallback done) const override
    {
        if (m_actualCtx) {
            return m_actualCtx->CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
        } else {
            return DeviceContext::CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
        }
    }

    void CopyDeviceTensorToCPU(const Tensor *device_tensor, StringPiece tensor_name,
                               Device *device, Tensor *cpu_tensor,
                               StatusCallback done) override
    {
        if (m_actualCtx) {
            return m_actualCtx->CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor, done);
        } else {
            return DeviceContext::CopyDeviceTensorToCPU(device_tensor, tensor_name, device,
                                                                    cpu_tensor, done);
        }
    }

    Device *device() const
    {
        return m_device;
    }

    DeviceContext *wrapped() const
    {
        return m_actualCtx;
    }

    WrapperFunction allocWrapper() const
    {
        return m_allocWrapper;
    }
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_WRAPPEDDEVICECONTEXT_H
