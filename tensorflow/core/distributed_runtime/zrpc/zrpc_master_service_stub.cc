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

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_master_service_stub.h"

#include "tensorflow/core/distributed_runtime/zrpc/protos/executor.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/platform/env.h"

#include <sstream>
#include <cstring>
#include <random>
#include <memory>

namespace zrpc = executor;
using std::ostringstream;
using random_bytes_engine = std::independent_bits_engine<std::random_device, sizeof(uint8_t), uint8_t>;

namespace tensorflow {

ZrpcMasterServiceStub::ZrpcMasterServiceStub(Env *env, const std::string &executorAddr)
    : m_execAddr(executorAddr)
    , m_zmqctx(1)
    , m_seq(0)
    , m_sendSock(m_zmqctx, zmq::socket_type::dealer)
{
    LOG(INFO) << "Created ZeroMQ context";

    random_bytes_engine rbe;
    m_recvId = "tensorflow::recv::";
    m_recvId += rbe();
    m_recvId += rbe();
    m_recvId += rbe();
    m_recvThread = env->StartThread(ThreadOptions(), "ZrpcMasterServiceStub::recvLoop",
                                    std::bind(&ZrpcMasterServiceStub::recvLoop, this));

    try {
        m_sendSock.connect(m_execAddr);
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "ZeroMQ socket connect failed: " << err.what();
    }
}

ZrpcMasterServiceStub::~ZrpcMasterServiceStub()
{
    // close socket before context, otherwise context close blocks
    m_sendSock.close();
    // close context before delete recv thread, so that recv thread will return.
    m_zmqctx.close();

    delete m_recvThread;
}

ZrpcMasterServiceStub::Item::Item() {}

ZrpcMasterServiceStub::Item::Item(Item &&other)
    : reply(std::move(other.reply))
    , done(std::move(other.done))
    , typedCallbacks(std::move(other.typedCallbacks))
{}

ZrpcMasterServiceStub::Item::Item(ProtoPtr &&rep, DoneCallback done) : reply(std::move(rep)), done(done) {}

ZrpcMasterServiceStub::Item &ZrpcMasterServiceStub::Item::operator=(Item &&other)
{
    std::swap(this->reply, other.reply);
    std::swap(this->done, other.done);
    std::swap(this->typedCallbacks, other.typedCallbacks);
    return *this;
}

ZrpcMasterServiceStub::Item::~Item()
{
    for (auto &item : typedCallbacks) {
        item.second(errors::Cancelled("Callback function cancelled"), nullptr);
    }
}

void ZrpcMasterServiceStub::dumpWaitingCb()
{
    mutex_lock locker(m_mtable);
    LOG(INFO) << "Pending callbacks:";
    for (auto &p : m_recvCallbacks) {
        auto &item = p.second;
        LOG(INFO) << "  seq: " << p.first;
        if (item.reply) {
            LOG(INFO) << "    reply type: " << item.reply->GetTypeName();
        } else {
            LOG(INFO) << "    reply type: nullptr";
        }
        LOG(INFO) << "   done: " << item.done.target_type().name();
        for (auto &typed : item.typedCallbacks) {
            LOG(INFO) << "    " << typed.first
                      << " -> " << typed.second.target_type().name();
        }
    }
}

void ZrpcMasterServiceStub::recvLoop()
{
    LOG(INFO) << "Started zmq recving thread, using ZMQ_IDENTITY: " << m_recvId;

    zmq::socket_t recvSock(m_zmqctx, zmq::socket_type::dealer);
    try {
        recvSock.setsockopt(ZMQ_IDENTITY, m_recvId.c_str(), m_recvId.size());
        recvSock.connect(m_execAddr);
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "ZeroMQ recving socket creation failed: " << err.what();
        return;
    }

    while (true) {
        try {
            dumpWaitingCb();
            // Receive and skip identification frames
            zmq::message_t msg;
            do {
                recvSock.recv(&msg);
            } while (msg.size() != 0 && recvSock.getsockopt<int64_t>(ZMQ_RCVMORE));

            // Now receive the message evenlop
            if (!recvSock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                LOG(ERROR) << "Skipped one iteration due to no evenlop message part found after identity frames";
                continue;
            }
            zmq::message_t msg_evenlop;
            recvSock.recv(&msg_evenlop);

            // Now receive the message body, don't skip if we can't find the body frame,
            // since we have the evenlop frame, we may find a callback to deal with this error.
            bool has_body = false;
            zmq::message_t msg_body;
            if (recvSock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                recvSock.recv(&msg_body);
                has_body = true;
            }

            // Parse what we have received
            zrpc::EvenlopDef edef;
            if(!edef.ParseFromArray(msg_evenlop.data(), msg_evenlop.size())) {
                LOG(ERROR) << "Received un-identifiable malformatted message. Dropping";
                continue;
            }

            LOG(INFO) << "Received evenlop: seq=" << edef.seq() << " type=" << edef.type();

            // Find corresonding item in table
            DoneCallback cb;
            ProtoPtr reply;
            {
                mutex_lock locker(m_mtable);
                auto it = m_recvCallbacks.find(edef.seq());
                if (it == m_recvCallbacks.end()) {
                    LOG(ERROR) << "Skipped one iteration due to seq not found in table: " << edef.seq();
                    continue;
                }
                auto itt = it->second.typedCallbacks.find(edef.type());
                if (itt != it->second.typedCallbacks.end()) {
                    cb = itt->second;
                    // we have a typed callback
                    if (!cb) {
                        LOG(WARNING) << "Skipped one iteration due to no callback";
                        continue;
                    }
                    auto desc = ::google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(edef.type());
                    if (!desc) {
                        LOG(ERROR) << "Protobuf descriptor not found for type name: " << edef.type();
                        cb(errors::Internal("Protobuf descriptor not found for type"), nullptr);
                        continue;
                    }
                    auto message = ::google::protobuf::MessageFactory::generated_factory()->GetPrototype(desc)->New();
                    if (!message) {
                        LOG(ERROR) << "Failed to create message object from descriptor of type name: {}" << edef.type();
                        cb(errors::Internal("Failed to create message object from descriptor of type"), nullptr);
                        continue;
                    }
                    reply.reset(message);
                } else {
                    if (!it->second.done) {
                        LOG(WARNING) << "Skipped one iteration due to no callback";
                        continue;
                    }
                    std::swap(cb ,it->second.done);
                    std::swap(reply ,it->second.reply);
                    auto t = it->second.typedCallbacks;
                    m_recvCallbacks.erase(it);
                }
            }

            // Now receive our message body
            if (reply && !has_body) {
                LOG(ERROR) << "Skipped one iteration due to no body message part found after evenlop frame";
                cb(errors::Internal("No body message found"), std::move(reply));
                continue;
            } else if (reply) {
                if(!reply->ParseFromArray(msg_body.data(), msg_body.size())) {
                    LOG(ERROR) << "Received malformatted message body. Dropping";
                    cb(errors::Internal("Body message malformatted"), std::move(reply));
                    continue;
                }
            }

            // Invoke callback regardless whether we have a body or not
            LOG(INFO) << "Calling callback function for seq " << edef.seq() << " and type " << edef.type();
            cb(Status::OK(), std::move(reply));
            LOG(INFO) << "Callback function returned for seq " << edef.seq() << " and type " << edef.type();
        } catch (zmq::error_t &err) {
            if (err.num() == ETERM || err.num() == EINTR) {
                break;
            }
            LOG(ERROR) << "Caught zmq error during recving loop: " << err.what();
        } catch (std::exception &err) {
            LOG(ERROR) << "Caught exception during recving loop: " << err.what();
        }
    }
}

template<typename ResponseType>
ZrpcMasterServiceStub::AsyncCallStarter ZrpcMasterServiceStub::rpcCallAsync(const std::string &sessionId,
                                                          const google::protobuf::Message& msg,
                                                          std::function<void(const Status&, std::unique_ptr<ResponseType>&&)> done)
{
    using ResponsePtr = std::unique_ptr<ResponseType>;
    Item args;
    if (done) {
        args = Item { ResponsePtr(new ResponseType), [done](const Status &s, ProtoPtr &&rep){
            done(s, ResponsePtr(static_cast<ResponseType*>(rep.release())));
        }};
    }

    return makeStarter(sessionId, msg, std::move(args));
}

ZrpcMasterServiceStub::AsyncCallStarter ZrpcMasterServiceStub::rpcCallAsync(const std::string &sessionId,
                                                          const ::google::protobuf::Message &msg)
{
    return makeStarter(sessionId, msg, {});
}

ZrpcMasterServiceStub::AsyncCallStarter ZrpcMasterServiceStub::makeStarter(const std::string &sessionId,
                                                         const ::google::protobuf::Message &msg,
                                                         Item &&cbitem)
{
    auto seq = m_seq.fetch_add(1);
    // Create evenlop message
    zrpc::EvenlopDef edef;
    edef.set_type(msg.GetTypeName());
    edef.set_seq(seq);
    if (!sessionId.empty()) {
        edef.set_sessionid(sessionId);
    }
    edef.set_recvidentity(m_recvId);
    edef.set_oplibrary(zrpc::TENSORFLOW);
    zmq::message_t evenlop(edef.ByteSizeLong());
    edef.SerializeToArray(evenlop.data(), evenlop.size());

    LOG(INFO) << "Sending evenlop message_t: " << edef.type() << " seq " << edef.seq();

    // Create body message
    zmq::message_t zmqmsg(msg.ByteSizeLong());
    msg.SerializeToArray(zmqmsg.data(), zmqmsg.size());

    if (cbitem.empty()) {
        return AsyncCallStarter(nullptr,
                                *this, seq, std::move(evenlop), std::move(zmqmsg));
    } else {
        mutex_lock locker(m_mtable);
        auto it = m_recvCallbacks.end();
        std::tie(it, std::ignore) = m_recvCallbacks.emplace(seq, std::move(cbitem));
        return AsyncCallStarter(&it->second.typedCallbacks,
                                *this, seq, std::move(evenlop), std::move(zmqmsg));
    }
}

void ZrpcMasterServiceStub::AsyncCallStarter::start()
{
    if (m_started) return;
    m_started = true;
    try {
        {
            mutex_lock locker(m_client.m_mu);
            m_client.m_sendSock.send(zmq::message_t(), ZMQ_SNDMORE);
            m_client.m_sendSock.send(m_evenlop, ZMQ_SNDMORE);
            m_client.m_sendSock.send(m_zmqmsg);
        }
        LOG(INFO) << "Message sent for seq: " << m_seq;
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "Error when sending message seq " << m_seq << ": " << err.what();
        // cleanup callback if any
        mutex_lock locker(m_client.m_mtable);
        m_client.m_recvCallbacks.erase(m_seq);
    }
}

template<typename ResponseType>
Status ZrpcMasterServiceStub::rpcCall(const std::string &sessionId, const ::google::protobuf::Message &msg,
                             std::unique_ptr<ResponseType> &reply)
{
    using ResponsePtr = std::unique_ptr<ResponseType>;

    Status status;
    Notification n;
    rpcCallAsync<ResponseType>(sessionId, msg, [&n, &status, &reply](const Status &s, ResponsePtr &&rep){
        status = s;
        reply = std::move(rep);
        n.Notify();
    });

    n.WaitForNotification();

    return status;
}

#define HANDLER_IMPL(name, sessIdExpr) \
Status ZrpcMasterServiceStub:: name (const name ## Request &req, name ## Response *resp) \
{ \
    LOG(INFO) << "==================================================================="; \
    LOG(INFO) << "RpcClient::" #name; \
\
    zrpc::CustomRequest request; \
    req.SerializeToString(request.mutable_extra()); \
    request.set_type(req.GetTypeName()); \
\
    std::unique_ptr<zrpc::CustomResponse> pResponse; \
    auto status = rpcCall((sessIdExpr), request, pResponse); \
\
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) { \
        LOG(ERROR) << "ZrpcMasterServiceStub::" << #name << " failed: " << status; \
        status.Update(errors::Internal("ZrpcMasterServiceStub::" #name " failed")); \
        return status; \
    } \
    if (!resp->ParseFromString(pResponse->extra())) { \
        LOG(ERROR) << "Response->extra is not a valid " #name "Response object"; \
        status.Update(errors::Internal("Response->extra is not a valid " #name "Response object")); \
        return status; \
    } \
\
    return Status::OK(); \
}

HANDLER_IMPL(ListDevices, "")
HANDLER_IMPL(Reset, "")
HANDLER_IMPL(CreateSession, "")

HANDLER_IMPL(CloseSession, req.session_handle())
HANDLER_IMPL(ExtendSession, req.session_handle())
HANDLER_IMPL(PartialRunSetup, req.session_handle())
HANDLER_IMPL(RunStep, req.session_handle())

#undef HANDLER_IMPL

} // namespace tensorflow
