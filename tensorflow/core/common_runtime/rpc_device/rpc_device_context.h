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

#ifndef RPCDEVICECONTEXT_H
#define RPCDEVICECONTEXT_H

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include <unordered_map>

namespace executor {
class OpKernelDef;
class OpContextDef;
}

namespace tensorflow {

class RPCDevice;
class RpcClient;

/**
 * @todo write docs
 */
class RPCDeviceContext : public DeviceContext
{
public:
    RPCDeviceContext(RPCDevice &device, RpcClient &client, const Graph *graph);

    ~RPCDeviceContext() override;

    void Compute(OpKernel *op_kernel, OpKernelContext *context);
    void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                      AsyncOpKernel::DoneCallback done);

    void serializeOpKernel(executor::OpKernelDef *def, OpKernel *kernel);
    void serializeOpContext(executor::OpContextDef *def, OpKernelContext *context);
    void deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def);

    Tensor tensorFromProtoMeta(const TensorProto &outdef);
    void tensorToProtoMeta(TensorProto *meta, const Tensor &tensor, bool is_ref);

    const std::string &execId() const;
    const std::string &sessionId() const;
    const GraphDef &graphDef() const;

private:
    const NodeDef &findNodeDefFor(const OpKernel *kernel) const;

private:
    RpcClient &m_rpc;
    RPCDevice &m_device;

    GraphDef m_graphdef;
    std::unordered_map<std::string, int> m_name2defidx;

    std::string m_execId;

    TF_DISALLOW_COPY_AND_ASSIGN(RPCDeviceContext);
};

}
#endif // RPCDEVICECONTEXT_H
