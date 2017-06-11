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

#include <memory>
#include <atomic>

namespace executor {
class OpKernelDef;
class OpContextDef;
}

namespace tensorflow {

class OpKernel;
class OpKernelContext;
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

    virtual void createSession(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph) = 0;

    using RunCallback = std::function<void()>;
    virtual void runAsync(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
                          OpKernel *kernel, OpKernelContext *context, RunCallback done) = 0;
    virtual Status run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
                       OpKernel *kernel, OpKernelContext *context) = 0;
    virtual Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) = 0;
    virtual Status deallocate(uint64_t addr_handle) = 0;
    virtual Status fetch(tensorflow::Tensor *cpu_tensor, const tensorflow::Tensor *dev_tensor) = 0;
    virtual Status push(tensorflow::Tensor *dev_tensor, const tensorflow::Tensor *cpu_tensor) = 0;

    void maybeInitialize(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph);

    void serializeOpKernel(executor::OpKernelDef *def, const OpKernel *kernel,
                           Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto);
    void serializeOpContext(executor::OpContextDef *def, OpKernelContext *context,
                            Graph *graph, const FunctionDefLibrary &library, const ConfigProto &cfgProto);
    void deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def);

private:
    std::atomic_flag m_initialized;

    TF_DISALLOW_COPY_AND_ASSIGN(RpcClient);
};

} // namespace tensorflow

#endif // RPCCLIENT_H
