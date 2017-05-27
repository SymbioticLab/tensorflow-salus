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

#include "tensorflow/core/common_runtime/rpc_device/rpc/zmqrpcclient.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.pb.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {

RpcClient::RpcClient() { }

RpcClient::~RpcClient() { }

RpcClient &RpcClient::instance()
{
    static ZmqRpcClient client;

    return client;
}

void RpcClient::serializeOpKernel(executor::OpKernelDef *def, const OpKernel *kernel)
{
    // TODO: serialize OpKernel to protobuf
    def->set_id(kernel->name());
    def->set_oplibrary(executor::OpKernelDef::TENSORFLOW);

    def->set_extra(kernel->def().SerializeAsString());
}

void RpcClient::serializeOpContext(executor::OpContextDef *def, const OpKernelContext *context)
{
    // TODO: serialize OpKernelContext to protobuf
}

void RpcClient::deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def)
{
    // TODO: deserialize OpKernelContext from protobuf
}

} // namespace tensorflow
