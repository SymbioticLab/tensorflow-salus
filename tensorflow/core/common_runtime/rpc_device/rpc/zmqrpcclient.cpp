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

#include "zmqrpcclient.h"

#include "tensorflow/core/common_runtime/rpc_device/rpc_device_context.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.pb.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/tfoplibrary.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

#include <sstream>
#include <cstring>
#include <random>
#include <memory>

namespace rpc = ::executor;
using std::ostringstream;
using random_bytes_engine = std::independent_bits_engine<std::random_device, sizeof(uint8_t), uint8_t>;

namespace tensorflow {

ZmqRpcClient::ZmqRpcClient(Env *env, const std::string &executorAddr)
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
    m_recvThread = env->StartThread(ThreadOptions(), "ZmqRpcClient::recvLoop",
                                    std::bind(&ZmqRpcClient::recvLoop, this));

    try {
        m_sendSock.connect(m_execAddr);
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "ZeroMQ socket connect failed: " << err.what();
    }
}

ZmqRpcClient::~ZmqRpcClient()
{
    // close socket before context, otherwise context close blocks
    m_sendSock.close();
    // close context before delete recv thread, so that recv thread will return.
    m_zmqctx.close();

    delete m_recvThread;
}

ZmqRpcClient::Item::Item() {}

ZmqRpcClient::Item::Item(Item &&other)
    : reply(std::move(other.reply))
    , done(std::move(other.done))
    , typedCallbacks(std::move(other.typedCallbacks))
{}

ZmqRpcClient::Item::Item(ProtoPtr &&rep, DoneCallback done) : reply(std::move(rep)), done(done) {}

ZmqRpcClient::Item &ZmqRpcClient::Item::operator=(Item &&other)
{
    std::swap(this->reply, other.reply);
    std::swap(this->done, other.done);
    std::swap(this->typedCallbacks, other.typedCallbacks);
    return *this;
}

void ZmqRpcClient::dumpWaitingCb()
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

void ZmqRpcClient::recvLoop()
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
            rpc::EvenlopDef edef;
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

ZmqRpcClient::Item::~Item()
{
    for (auto &item : typedCallbacks) {
        item.second(errors::Cancelled("Callback function cancelled"), nullptr);
    }
}

template<typename ResponseType>
ZmqRpcClient::AsyncCallStarter ZmqRpcClient::rpcCallAsync(const std::string &sessionId,
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

ZmqRpcClient::AsyncCallStarter ZmqRpcClient::rpcCallAsync(const std::string &sessionId,
                                                          const ::google::protobuf::Message &msg)
{
    return makeStarter(sessionId, msg, {});
}

ZmqRpcClient::AsyncCallStarter ZmqRpcClient::makeStarter(const std::string &sessionId,
                                                         const ::google::protobuf::Message &msg,
                                                         Item &&cbitem)
{
    auto seq = m_seq.fetch_add(1);
    // Create evenlop message
    rpc::EvenlopDef edef;
    edef.set_type(msg.GetTypeName());
    edef.set_seq(seq);
    if (!sessionId.empty()) {
        edef.set_sessionid(sessionId);
    }
    edef.set_recvidentity(m_recvId);
    edef.set_oplibrary(executor::TENSORFLOW);
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

void ZmqRpcClient::AsyncCallStarter::start()
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
Status ZmqRpcClient::rpcCall(const std::string &sessionId, const ::google::protobuf::Message &msg,
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

void ZmqRpcClient::createSession(const ConfigProto & cfgProto, std::string &sessionId)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::createSession";
    rpc::TFSessionArgs args;
    *args.mutable_cfgproto() = cfgProto;

    rpc::CustomRequest request;
    args.SerializeToString(request.mutable_extra());
    request.set_type(args.GetTypeName());

    std::unique_ptr<rpc::CustomResponse> pResponse;
    auto status = rpcCall("", request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        LOG(ERROR) << "RpcClient::createSession failed";
        return;
    }

    rpc::TFSessionCreated sesscreated;
    if (!sesscreated.ParseFromString(pResponse->extra())) {
        LOG(ERROR) << "Response->extra is not a valid TFSessionCreated object";
        return;
    }

    sessionId = sesscreated.sessionid();
    LOG(INFO) << "RpcClient created session with id " << sessionId;
}

void ZmqRpcClient::closeSession(const std::string &sessionId)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::closeSession";
    rpc::TFSessionClose tfclose;
    tfclose.set_sessionid(sessionId);

    rpc::CustomRequest request;
    tfclose.SerializeToString(request.mutable_extra());
    request.set_type(tfclose.GetTypeName());

    std::unique_ptr<rpc::CustomResponse> pResponse;
    auto status = rpcCall(sessionId, request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        LOG(ERROR) << "RpcClient::closeSession failed";
        return;
    }
}

void ZmqRpcClient::execSetup(RPCDeviceContext *devCtx, std::string &execId)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::execSetup";
    rpc::RunGraphRequest request;
    auto computation = request.mutable_computation();
    devCtx->graphDef().SerializeToString(computation->mutable_extra());
    computation->set_oplibrary(rpc::TENSORFLOW);

    std::unique_ptr<rpc::RunGraphResponse> pResponse;
    auto status = rpcCall(devCtx->sessionId(), request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        LOG(ERROR) << "RpcClient::execSetup failed";
        return;
    }

    execId = pResponse->execid();
    LOG(INFO) << "RpcClient created exec with id " << execId;
}

void ZmqRpcClient::runAsync(RPCDeviceContext *devCtx, AsyncOpKernel *kernel, OpKernelContext *context,
                            AsyncOpKernel::DoneCallback done)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::runAsync";

    rpc::RunRequest request;
    devCtx->serializeOpKernel(request.mutable_opkernel(), kernel);
    devCtx->serializeOpContext(request.mutable_context(), context);
    request.set_execid(devCtx->execId());

    Item args;
    LOG(INFO) << "RpcClient::runAsync    calling rpc using rpc stub";
    using ResponsePtr = std::unique_ptr<rpc::RunResponse>;
    auto starter = rpcCallAsync<rpc::RunResponse>(devCtx->sessionId(), request,
                                                  [done, devCtx, context](const Status &status,
                                                                          ResponsePtr &&pResponse) {
        LOG(INFO) << "RpcClient::runAsync    rpc returned with status: "
                  << status.code() << " " << status.error_message();

        // TODO: better error handling
        if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
            return;
        }

        devCtx->deserializeOpContext(context, &pResponse->context());
        done();
    });

    auto seq = starter.seq();
    auto sessionId = devCtx->sessionId();
    starter.add("executor.TFRendezRecvRequests", [seq, sessionId, context, this](const Status &s, ProtoPtr &&msg){
        if (!s.ok()) {
            return;
        }
        std::unique_ptr<executor::TFRendezRecvRequests> pReq;
        pReq.reset(static_cast<executor::TFRendezRecvRequests*>(msg.release()));
        for (size_t i = 0; i != pReq->key_size(); ++i) {
            Rendezvous::ParsedKey parsed;
            auto s = context->rendezvous()->ParseKey(pReq->key(i), &parsed);
            if (!s.ok()) {
                LOG(ERROR) << "Invalid rendezvous key in TFRendezRecvRequests: " << pReq->key(i);
                continue;
            }
            LOG(INFO) << "Got executor.TFRendezRecvRequests for " << pReq->key(i);
            Rendezvous::Args args;
            args.alloc_attrs.value = pReq->allocattributes(i);
            args.alloc_attrs.set_on_host(true); // we want tensor to on CPU, because we send them out ourselves
            args.device_context = context->op_device_context();
            context->rendezvous()->RecvAsync(parsed, args,
                                            [this, seq, sessionId, parsed](const Status &s,
                                                                        const Rendezvous::Args &send_args,
                                                                        const Rendezvous::Args &recv_args,
                                                                        const Tensor &val, bool is_dead){
                LOG(INFO) << "Send out executor.TFRendezRecvUpdate for " << parsed.FullKey();
                rpc::TFRendezRecvUpdate resp;
                resp.set_forseq(seq);
                auto item = resp.add_items();
                item->set_key(parsed.FullKey().ToString());
    //             item->set_allocattributes(send_args.alloc_attrs.value);
                val.AsProtoTensorContent(item->mutable_val());

                rpc::CustomRequest request;
                request.set_type(resp.GetTypeName());
                resp.SerializeToString(request.mutable_extra());
                this->rpcCallAsync(sessionId, request);
                LOG(INFO) << "Send out executor.TFRendezRecvUpdate finish " << parsed.FullKey();
            });
            LOG(INFO) << "Called rendezvous recvasync for " << parsed.FullKey();
        }
    });
}

Status ZmqRpcClient::run(RPCDeviceContext *devCtx, OpKernel *kernel, OpKernelContext *context)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::run";

    rpc::RunRequest request;
    devCtx->serializeOpKernel(request.mutable_opkernel(), kernel);
    devCtx->serializeOpContext(request.mutable_context(), context);
    request.set_execid(devCtx->execId());

    std::unique_ptr<rpc::RunResponse> pResponse;
    LOG(INFO) << "RpcClient::run    calling rpc using rpc stub";
    auto status = rpcCall(devCtx->sessionId(), request, pResponse);
    LOG(INFO) << "RpcClient::run    rpc returned with status: "
              << status.code() << " " << status.error_message();

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    devCtx->deserializeOpContext(context, &pResponse->context());

    return context->status();
}

Status ZmqRpcClient::allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle)
{
    LOG(INFO) << "RpcClient::allocate(alignment=" << alignment << ", num_bytes=" << num_bytes << ")";

    rpc::AllocRequest request;
    request.set_alignment(alignment);
    request.set_num_bytes(num_bytes);

    std::unique_ptr<rpc::AllocResponse> pResponse;
    auto status = rpcCall("", request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    *addr_handle = pResponse->addr_handle();
    LOG(INFO) << "RpcClient::allocate returned addr_handle=" << addr_handle;
    return Status::OK();
}

Status ZmqRpcClient::deallocate(uint64_t addr_handle)
{
    LOG(INFO) << "RpcClient::deallocate(addr_handle=" << addr_handle;

    rpc::DeallocRequest request;
    request.set_addr_handle(addr_handle);

    std::unique_ptr<rpc::DeallocResponse> pResponse;
    auto status = rpcCall("", request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    return Status::OK();
}
} // namespace tensorflow
