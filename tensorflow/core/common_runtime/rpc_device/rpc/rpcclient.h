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

#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.pb.h"

#include "zmq.hpp"

#include <memory>

namespace tensorflow {

class OpKernel;
class OpKernelContext;

/**
 * @todo write docs
 */
class RpcClient
{
public:
    RpcClient();

    virtual ~RpcClient();

    virtual Status run(OpKernel *kernel, OpKernelContext *context) = 0;
    virtual Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) = 0;
    virtual Status deallocate(uint64_t addr_handle) = 0;

    // default instance always connect to localhost:55001
    static RpcClient &instance();

private:
//     RpcClient(std::shared_ptr<grpc::Channel> channel);

//     std::unique_ptr<executor::IExecEngine::Stub> m_stub;
};

class ZmqRpcClient : public RpcClient
{
public:
    ZmqRpcClient();

    ~ZmqRpcClient() override;

    Status run(OpKernel *kernel, OpKernelContext *context) override;
    Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) override;
    Status deallocate(uint64_t addr_handle) override;

private:
    Status rpcCall(::google::protobuf::Message &msg, ::google::protobuf::Message &reply);

private:
    zmq::context_t m_zmqctx;
    zmq::socket_t m_zmqsock;
};

} // namespace tensorflow

#endif // RPCCLIENT_H
