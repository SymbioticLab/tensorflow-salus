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

#include "rpc_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
RPCDevice::RPCDevice(const SessionOptions &options, const string &name, Bytes memory_limit,
                     const DeviceLocality &locality, Allocator *allocator, RpcClient &rpc)
    : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_RPC, memory_limit, locality), allocator)
    , m_allocator(allocator)
    , m_rpc(rpc)
    , m_cfgProto(options.config)
{
}

RPCDevice::~RPCDevice()
{

}

Status RPCDevice::Sync()
{
    return Status::OK();
}

Status RPCDevice::MaybeRewriteGraph(const FunctionDefLibrary& library, std::unique_ptr<Graph>* graph)
{
    m_funcDefLib = library;

    m_graph = (*graph).get();

    return Status::OK();
}

Status RPCDevice::FillContextMap(const Graph* graph, DeviceContextMap* device_context_map)
{
    LOG(INFO) << "RpcDevice::FillContextMap";
    device_context_map->resize(graph->num_node_ids());
    auto* ctx = new RPCDeviceContext(m_rpc);
    for (Node* n : graph->nodes()) {
        LOG(INFO) << n->id() << " : " << n->type_string() << " : " << n->name();
        ctx->Ref();
        (*device_context_map)[n->id()] = ctx;
    }
    ctx->Unref();
    return Status::OK();
}

void RPCDevice::Compute(OpKernel *op_kernel, OpKernelContext *context)
{
    auto status = m_rpc.run(m_cfgProto, m_funcDefLib, m_graph, op_kernel, context);

    if (!status.ok()) {
        LOG(ERROR) << "RPC call failed with " << status;
    }

//     op_kernel->Compute(context);

    LOG(INFO) << "context.status() " << context->status();
    LOG(INFO) << "context.is_output_dead() " << *context->is_output_dead();
    LOG(INFO) << "context.num_outputs() " << context->num_outputs();
}

void RPCDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context, AsyncOpKernel::DoneCallback done)
{
    m_rpc.runAsync(m_cfgProto, m_funcDefLib, m_graph, op_kernel, context, std::move(done));
//     op_kernel->ComputeAsync(context, std::move(done));
}

Allocator *RPCDevice::GetAllocator(AllocatorAttributes attr)
{
    return m_allocator;
}

Status RPCDevice::MakeTensorFromProto(const TensorProto &tensor_proto, const AllocatorAttributes alloc_attrs,
                                      Tensor *tensor)
{
    LOG(WARNING) << "!!!RpcDevice MakeTensorFromProto";

    // TODO: implement make tensor from proto through rpc
    if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
        Tensor parsed(tensor_proto.dtype());
        if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
            *tensor = parsed;
            return Status::OK();
        }
    }
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
}

} // namespace tensorflow
