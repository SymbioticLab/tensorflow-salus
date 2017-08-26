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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_MASTER_SERVICE_STUB_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_MASTER_SERVICE_STUB_H_

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_util.h"
#include "tensorflow/core/distributed_runtime/zrpc/protos/executor.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

#include "zmq.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>

namespace tensorflow {
class Env;
class Thread;

class CreateSessionRequest;
class CreateSessionResponse;
class ExtendSessionRequest;
class ExtendSessionResponse;
class PartialRunSetupRequest;
class PartialRunSetupResponse;
class RunStepRequest;
class RunStepResponse;
class CloseSessionRequest;
class CloseSessionResponse;
class ListDevicesRequest;
class ListDevicesResponse;
class ResetRequest;
class ResetResponse;

using ProtoPtr = std::unique_ptr<::google::protobuf::Message>;

class ZrpcMasterServiceStub
{
public:
    ZrpcMasterServiceStub(Env *env, const std::string &executorAddr);

    ~ZrpcMasterServiceStub();

    Status CreateSession(const CreateSessionRequest &req, CreateSessionResponse *resp);
    Status CloseSession(const CloseSessionRequest &req, CloseSessionResponse *resp);
    Status ExtendSession(const ExtendSessionRequest &req, ExtendSessionResponse *resp);
    Status PartialRunSetup(const PartialRunSetupRequest &req, PartialRunSetupResponse *resp);
    Status RunStep(const RunStepRequest &req, RunStepResponse *resp);
    Status ListDevices(const ListDevicesRequest &req, ListDevicesResponse *resp);
    Status Reset(const ResetRequest &req, ResetResponse *resp);

private:
    using DoneCallback = std::function<void(const Status &, ProtoPtr &&)>;
    template<typename ResponseType>
    Status rpcCall(const std::string &sessionId, const ::google::protobuf::Message &msg,
                   std::unique_ptr<ResponseType> &pReply);

    struct AsyncCallStarter
    {
        AsyncCallStarter(const AsyncCallStarter &other) = delete;
        AsyncCallStarter &operator=(const AsyncCallStarter &other) = delete;

        AsyncCallStarter(AsyncCallStarter &&other)
            : m_pTypedCallbacks(other.m_pTypedCallbacks)
            , m_client(other.m_client)
            , m_seq(other.m_seq)
            , m_evenlop(std::move(other.m_evenlop))
            , m_zmqmsg(std::move(other.m_zmqmsg))
            , m_started(std::move(other.m_started))
        {
            other.m_pTypedCallbacks = nullptr;
            other.m_started = true;
        }

        AsyncCallStarter(std::unordered_map<std::string, DoneCallback> *pTypedCallbacks, ZrpcMasterServiceStub &client,
                         uint64_t seq, zmq::message_t &&evenlop, zmq::message_t &&zmqmsg)
            : m_pTypedCallbacks(pTypedCallbacks)
            , m_client(client)
            , m_seq(seq)
            , m_evenlop(std::move(evenlop))
            , m_zmqmsg(std::move(zmqmsg))
            , m_started(false)
        {
        }

        ~AsyncCallStarter()
        {
            start();
        }

        void start();

        bool add(const std::string &type, DoneCallback cb)
        {
            if (m_pTypedCallbacks) {
                (*m_pTypedCallbacks)[type] = std::move(cb);
                return true;
            }
            LOG(WARNING) << "Adding typed callback to an empty (null) starter";
            return false;
        }

        uint64_t seq() const
        {
            return m_seq;
        };

    private:
        std::unordered_map<std::string, DoneCallback> *m_pTypedCallbacks;
        ZrpcMasterServiceStub &m_client;
        uint64_t m_seq;
        zmq::message_t m_evenlop;
        zmq::message_t m_zmqmsg;
        bool m_started;
    };

    template<typename ResponseType>
    AsyncCallStarter rpcCallAsync(const std::string &sessionId, const ::google::protobuf::Message &msg,
                                  std::function<void(const Status &, std::unique_ptr<ResponseType> &&)> done);

    AsyncCallStarter rpcCallAsync(const std::string &sessionId, const ::google::protobuf::Message &msg);

    void recvLoop();
    void dumpWaitingCb();

    struct Item;
    AsyncCallStarter makeStarter(const std::string &sessionId, const ::google::protobuf::Message &msg,
                                 Item &&cbitem);

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

        ~Item();
        bool empty() const
        {
            return !reply && !done && typedCallbacks.empty();
        }

        Item();
        Item(Item &&other);
        Item(ProtoPtr &&rep, DoneCallback d);
        Item &operator=(Item &&other);
    };

    mutex m_mtable;
    std::unordered_map<uint64_t, Item> m_recvCallbacks GUARDED_BY(m_mtable);
    std::string m_recvId;
    Thread *m_recvThread;

    TF_DISALLOW_COPY_AND_ASSIGN(ZrpcMasterServiceStub);
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_MASTER_SERVICE_STUB_H_
