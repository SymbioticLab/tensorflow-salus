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
    // TODO: serialize OpKernel to protobuf
    LOG(INFO) << "About to serialize OpKernel";

    LOG(INFO) << "def " << def;
    LOG(INFO) << "kernel " << kernel;
    LOG(INFO) << "graph " << graph;

    def->set_id(kernel->name());
    def->set_oplibrary(executor::OpKernelDef::TENSORFLOW);
    def->set_graph_def_version(graph->versions().producer());

    LOG(INFO) << "Creating coded output stream";
    google::protobuf::io::StringOutputStream raw_output(def->mutable_extra());
    google::protobuf::io::CodedOutputStream coded_stream(&raw_output);

    LOG(INFO) << "Writing NodeDef";
    const auto &ndef = kernel->def();
    coded_stream.WriteVarint32(ndef.ByteSize());
    ndef.SerializeToCodedStream(&coded_stream);

    LOG(INFO) << "Writing FunctionDefLibrary";
    LOG(INFO) << "library byte size " << library.ByteSize();
    LOG(INFO) << "library content " << library.DebugString();
    coded_stream.WriteVarint32(library.ByteSize());

    LOG(INFO) << "library size written";
    auto ok = library.SerializeToCodedStream(&coded_stream);
    if (!ok) {
        LOG(ERROR) << "library serialize to coded stream failed";
    }

    LOG(INFO) << "Writing ConfigProto";
    LOG(INFO) << "cfgProto byte size " << cfgProto.ByteSize();
    LOG(INFO) << "cfgProto content " << cfgProto.DebugString();
    coded_stream.WriteVarint32(cfgProto.ByteSize());
    cfgProto.SerializeToCodedStream(&coded_stream);

    LOG(INFO) << "Done";
}

void RpcClient::serializeOpContext(executor::OpContextDef *def, const OpKernelContext *context,
                                   Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto)
{
    // TODO: serialize OpKernelContext to protobuf
}

void RpcClient::deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def)
{
    // TODO: deserialize OpKernelContext from protobuf
}

} // namespace tensorflow
