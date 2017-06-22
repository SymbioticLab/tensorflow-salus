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

#ifndef RPCCLIENT_H
#define RPCCLIENT_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/graph.pb.h"

#include <memory>
#include <atomic>
#include <functional>
#include <unordered_map>


using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;

namespace executor {
class OpKernelDef;
class OpContextDef;
}

namespace tensorflow {

class Graph;
class FunctionDefLibrary;
class ConfigProto;
class Tensor;

/**
 * @todo write docs
 */
class RpcClient
{
public:
    RpcClient();

    virtual ~RpcClient();

    using DoneCallback = std::function<void(const Status&, ProtoPtr&&)>;

    virtual void createSession(const ConfigProto &cfgProto, const FunctionDefLibrary &library,
                               const GraphDef &graphdef) = 0;

    virtual void runAsync(const ConfigProto &cfgProto, const FunctionDefLibrary &library, const Graph *graph,
                          AsyncOpKernel *kernel, OpKernelContext *context, AsyncOpKernel::DoneCallback done) = 0;
    virtual Status run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, const Graph *graph,
                       OpKernel *kernel, OpKernelContext *context) = 0;
    virtual Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) = 0;
    virtual Status deallocate(uint64_t addr_handle) = 0;

    void maybeInitialize(const ConfigProto &cfgProto, const FunctionDefLibrary &library, const Graph *graph);

    void serializeOpKernel(executor::OpKernelDef *def, OpKernel *kernel,
                           const Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto);
    void serializeOpContext(executor::OpContextDef *def, OpKernelContext *context,
                            const Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto);
    void deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def);

protected:
    Tensor tensorFromProtoMeta(const TensorProto &outdef);
    void tensorToProtoMeta(TensorProto *meta, const Tensor &tensor, bool is_ref);

private:
    std::atomic_flag m_initialized;
    GraphDef m_graphdef;
    std::unordered_map<std::string, int> m_name2defidx;

    TF_DISALLOW_COPY_AND_ASSIGN(RpcClient);
};

} // namespace tensorflow

#endif // RPCCLIENT_H
