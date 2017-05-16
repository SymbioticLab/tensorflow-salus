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

#ifndef RPC_DEVICE_H
#define RPC_DEVICE_H

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_allocator.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_device_context.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

/**
 * @todo write docs
 */
class RpcDevice : public LocalDevice
{
public:

    RpcDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
              const DeviceLocality &locality, Allocator *allocator);

    ~RpcDevice() override;

    void Compute(OpKernel *op_kernel, OpKernelContext *context) override;
    Allocator *GetAllocator(AllocatorAttributes attr) override;
    Status MakeTensorFromProto(const TensorProto &tensor_proto, const AllocatorAttributes alloc_attrs,
                               Tensor *tensor) override;

    Status Sync() override;

private:
    Allocator *m_allocator;  // Not owned
};

}
#endif // RPC_DEVICE_H
