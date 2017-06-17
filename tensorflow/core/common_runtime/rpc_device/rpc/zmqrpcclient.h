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
    ZmqRpcClient(Env *env, const std::string &executorAddr);

    ~ZmqRpcClient() override;

    void createSession(const ConfigProto & cfgProto, const FunctionDefLibrary & library, Graph *graph) override;

    void runAsync(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
                  AsyncOpKernel *kernel, OpKernelContext *context, AsyncOpKernel::DoneCallback done) override;

    Status run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
               OpKernel *kernel, OpKernelContext *context) override;
    Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) override;
    Status deallocate(uint64_t addr_handle) override;
    Status fetch(tensorflow::Tensor *cpu_tensor, const tensorflow::Tensor *dev_tensor) override;
    Status push(tensorflow::Tensor *dev_tensor, const tensorflow::Tensor *cpu_tensor) override;
private:
    using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;
    template<typename ResponseType>
    Status rpcCall(const ::google::protobuf::Message &msg, std::unique_ptr<ResponseType> &pReply);

    struct AsyncCallStarter
    {
        AsyncCallStarter(const AsyncCallStarter &other) = delete;
        AsyncCallStarter &operator=(const AsyncCallStarter &other) = delete;

        AsyncCallStarter(AsyncCallStarter &&other)
            : m_typedCallbacks(other.m_typedCallbacks)
            , m_client(other.m_client)
            , m_seq(other.m_seq)
            , m_evenlop(std::move(other.m_evenlop))
            , m_zmqmsg(std::move(other.m_zmqmsg))
            , m_started(std::move(other.m_started))
        {
            other.m_started = true;
        }

        AsyncCallStarter(std::unordered_map<std::string, DoneCallback> &typedCallbacks,
                         ZmqRpcClient &client, uint64_t seq, zmq::message_t &&evenlop, zmq::message_t &&zmqmsg)
            : m_typedCallbacks(typedCallbacks)
            , m_client(client)
            , m_seq(seq)
            , m_evenlop(std::move(evenlop))
            , m_zmqmsg(std::move(zmqmsg))
            , m_started(false)
        {}

        ~AsyncCallStarter()
        {
            start();
        }

        void start();

        void add(const std::string &type, DoneCallback cb)
        {
            m_typedCallbacks[type] = std::move(cb);
        }
    private:
        std::unordered_map<std::string, DoneCallback> &m_typedCallbacks;
        ZmqRpcClient &m_client;
        uint64_t m_seq;
        zmq::message_t m_evenlop;
        zmq::message_t m_zmqmsg;
        bool m_started;
    };

    template<typename ResponseType>
    AsyncCallStarter rpcCallAsync(const ::google::protobuf::Message &msg,
                                  std::function<void(const Status&, std::unique_ptr<ResponseType>&&)> done);

    AsyncCallStarter rpcCallAsync(const ::google::protobuf::Message &msg);
    void recvLoop();

private:
    std::string m_execAddr;

    zmq::context_t m_zmqctx;

    std::atomic<uint64_t> m_seq;

    mutex m_mu;
    zmq::socket_t m_sendSock GUARDED_BY(m_mu);

    struct Item
    {
        ProtoPtr reply;
        DoneCallback done;
        std::unordered_map<std::string, DoneCallback> typedCallbacks;

        Item();
        Item(Item &&other);
        Item(ProtoPtr &&rep, DoneCallback d);
        Item &operator=(Item &&other);
    };

    mutex m_mtable;
    std::unordered_map<uint64_t, Item> m_recvCallbacks GUARDED_BY(m_mtable);
    std::string m_recvId;
    Thread *m_recvThread;

    TF_DISALLOW_COPY_AND_ASSIGN(ZmqRpcClient);
};

} // namespace tensorflow

#endif // ZMQRPCCLIENT_H
