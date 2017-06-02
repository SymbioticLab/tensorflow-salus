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

#include "rpcclient.h"

#include "tensorflow/core/common_runtime/rpc_device/rpc_allocator.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/zmqrpcclient.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.pb.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/tfoplibrary.pb.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

RpcClient::RpcClient() { }

RpcClient::~RpcClient() { }

RpcClient &RpcClient::instance()
{
    static ZmqRpcClient client;

    return client;
}

void RpcClient::serializeOpKernel(executor::OpKernelDef *def, const tensorflow::OpKernel *kernel,
                                  Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto)
{
    LOG(INFO) << "About to serialize OpKernel";

    LOG(INFO) << "def " << def;
    LOG(INFO) << "kernel " << kernel;
    LOG(INFO) << "graph " << graph;

    def->set_id(kernel->name());
    def->set_oplibrary(executor::TENSORFLOW);

    executor::TFOpKernelDef tfdef;
    tfdef.set_graph_def_version(graph->versions().producer());

    *tfdef.mutable_nodedef() = kernel->def();
    *tfdef.mutable_cfgproto() = cfgProto;
    *tfdef.mutable_funcdef() = library;

    tfdef.SerializeToString(def->mutable_extra());

    LOG(INFO) << "Done";
}

void RpcClient::serializeOpContext(executor::OpContextDef *def, OpKernelContext *context,
                                   Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto)
{
    LOG(INFO) << "About to serialize OpContext";

    executor::TFOpContextDef tfdef;
    auto params = context->params_;

    tfdef.set_step_id(params->step_id);
    tfdef.set_frame_id(params->frame_iter.frame_id);
    tfdef.set_iter_id(params->frame_iter.iter_id);
    tfdef.set_is_input_dead(params->is_input_dead);

    for (int i = 0; i != context->num_inputs(); i++) {
        auto in = context->input(i);
        auto indef = tfdef.add_inputs();

        indef->set_dtype(in.dtype());
        in.shape().AsProto(indef->mutable_tensor_shape());

        auto addr_handle = reinterpret_cast<uint64_t>(in.tensor_data().data());
        // HACK: use a int64 val entry to store the addr handle for simplicity,
        // idealy should store this in tensor_content with proper encoding.
        indef->add_int64_val(addr_handle);
    }

    tfdef.SerializeToString(def->mutable_extra());

    LOG(INFO) << "Done";
}

void RpcClient::deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def)
{
    LOG(INFO) << "About to update context";

    executor::TFOpContextUpdate tfdef;
    tfdef.ParseFromString(def->extra());

    *context->is_output_dead() = tfdef.is_output_dead();

    if (tfdef.status_code() == error::OK) {
        context->SetStatus(Status::OK());
    } else {
        context->SetStatus(Status(tfdef.status_code(), tfdef.status_msg()));
    }

    for (int i = 0; i != tfdef.outputs_size(); ++i) {
        const auto &outdef = tfdef.outputs(i);
        // create a one time allocator, it will delete itself after DeallocateRaw
        auto alloc = OneTimeAllocator::create(outdef.int64_val(0)).release();
        Tensor output(alloc, outdef.dtype(), TensorShape(outdef.tensor_shape()));
        context->set_output(i, output);
    }

    LOG(INFO) << "Done";
}

} // namespace tensorflow
