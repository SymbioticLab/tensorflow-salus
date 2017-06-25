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

#include <memory>
#include <functional>

using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;

namespace tensorflow {

class FunctionDefLibrary;
class ConfigProto;
class GraphDef;
class RPCDeviceContext;

/**
 * @todo write docs
 */
class RpcClient
{
public:
    RpcClient();

    virtual ~RpcClient();

    using DoneCallback = std::function<void(const Status&, ProtoPtr&&)>;

    virtual void createSession(const ConfigProto &cfgProto, std::string &sessionId) = 0;
    virtual void closeSession(const std::string &sessionId) = 0;

    virtual void execSetup(RPCDeviceContext *devCtx, std::string &execId) = 0;

    virtual void runAsync(RPCDeviceContext *devCtx, AsyncOpKernel *kernel, OpKernelContext *context,
                          AsyncOpKernel::DoneCallback done) = 0;
    virtual Status run(RPCDeviceContext *devCtx, OpKernel *kernel, OpKernelContext *context) = 0;
    virtual Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) = 0;
    virtual Status deallocate(uint64_t addr_handle) = 0;

private:
    TF_DISALLOW_COPY_AND_ASSIGN(RpcClient);
};

} // namespace tensorflow

#endif // RPCCLIENT_H
