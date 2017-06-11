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

#ifndef ZMQRPCCLIENT_H
#define ZMQRPCCLIENT_H

#include "tensorflow/core/common_runtime/rpc_device/rpc/rpcclient.h"
#include "tensorflow/core/platform/mutex.h"

#include "zmq.hpp"

#include <atomic>
#include <unordered_map>
#include <memory>

namespace tensorflow {
class Env;
class Thread;

class ZmqRpcClient : public RpcClient
{
public:
    ZmqRpcClient(Env *env);

    ~ZmqRpcClient() override;

    Status run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
               OpKernel *kernel, OpKernelContext *context) override;
    Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) override;
    Status deallocate(uint64_t addr_handle) override;
    Status fetch(tensorflow::Tensor *cpu_tensor, const tensorflow::Tensor *dev_tensor) override;
    Status push(tensorflow::Tensor *dev_tensor, const tensorflow::Tensor *cpu_tensor) override;

private:
    using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;
    using DoneCallback = std::function<void(const Status&, ProtoPtr&&)>;
    struct Args
    {
        ProtoPtr reply;
        DoneCallback done;

        Args();
        Args(Args &&other);
        Args(ProtoPtr &&rep, DoneCallback d);
        Args &operator=(Args &&other);
    };

    template<typename ResponseType>
    Status rpcCall(const ::google::protobuf::Message &msg, std::unique_ptr<ResponseType> &pReply);

    template<typename ResponseType>
    void rpcCallAsync(const ::google::protobuf::Message &msg, Args &&args);

    void recvLoop();

private:
    zmq::context_t m_zmqctx;

    std::atomic_uint64_t m_seq;

    mutex m_mu;
    zmq::socket_t m_sendSock GUARDED_BY(m_mu);

    mutex m_mtable;
    std::unordered_map<uint64_t, Args> m_recvCallbacks GUARDED_BY(m_mtable);
    std::string m_recvId;
    Thread *m_recvThread;

    TF_DISALLOW_COPY_AND_ASSIGN(ZmqRpcClient);
};

} // namespace tensorflow

#endif // ZMQRPCCLIENT_H
