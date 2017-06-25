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

#include "rpc/rpcclient.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_allocator.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_device_context.h"
#include "tensorflow/core/public/session_options.h"

#include <memory>

namespace tensorflow {

/**
 * @todo write docs
 */
// TODO: derive from Device directly
class RPCDevice : public LocalDevice
{
public:

    RPCDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
              const DeviceLocality &locality, Allocator *allocator, RpcClient &rpc);

    ~RPCDevice() override;

    void Compute(OpKernel *op_kernel, OpKernelContext *context) override;
    void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                      AsyncOpKernel::DoneCallback done) override;

    Allocator *GetAllocator(AllocatorAttributes attr) override;
    Status MakeTensorFromProto(const TensorProto &tensor_proto, const AllocatorAttributes alloc_attrs,
                               Tensor *tensor) override;

    Status Sync() override;
    Status MaybeRewriteGraph(const FunctionDefLibrary& library, std::unique_ptr<Graph>* graph) override;

    Status FillContextMap(const Graph* graph, DeviceContextMap* device_context_map) override;

    const std::string &sessionId() const;
private:
    Allocator *m_allocator;  // Not owned

    RpcClient &m_rpc;

    std::string m_sessionId;

    ConfigProto m_cfgProto;
};

}
#endif // RPC_DEVICE_H
