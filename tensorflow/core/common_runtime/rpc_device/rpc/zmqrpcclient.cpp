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
            zmq::message_t msg;
            // Receive and skip identification frames
            do {
                recvSock.recv(&msg);
            } while (msg.size() != 0 && recvSock.getsockopt<int64_t>(ZMQ_RCVMORE));

            // Now receive our message evenlop
            if (!recvSock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                LOG(ERROR) << "Skipped one iteration due to no evenlop message part found after identity frames";
                continue;
            }
            recvSock.recv(&msg);
            rpc::EvenlopDef edef;
            if(!edef.ParseFromArray(msg.data(), msg.size())) {
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
                    m_recvCallbacks.erase(it);
                }
            }

            // Now receive our message body
            if (reply) {
                if (!recvSock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                    LOG(ERROR) << "Skipped one iteration due to no body message part found after identity frames";
                    cb(errors::Internal("No body message found"), std::move(reply));
                    continue;
                }
                recvSock.recv(&msg);
                if(!reply->ParseFromArray(msg.data(), msg.size())) {
                    LOG(ERROR) << "Received malformatted message body. Dropping";
                    cb(errors::Internal("Body message malformatted"), std::move(reply));
                    continue;
                }
            }
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
ZmqRpcClient::AsyncCallStarter ZmqRpcClient::rpcCallAsync(const google::protobuf::Message& msg,
                                                          std::function<void(const Status&, std::unique_ptr<ResponseType>&&)> done)
{
    auto seq = m_seq.fetch_add(1);
    // Create evenlop message
    rpc::EvenlopDef edef;
    edef.set_type(msg.GetTypeName());
    edef.set_seq(seq);
    edef.set_recvidentity(m_recvId);
    edef.set_oplibrary(executor::TENSORFLOW);
    zmq::message_t evenlop(edef.ByteSizeLong());
    edef.SerializeToArray(evenlop.data(), evenlop.size());

    LOG(INFO) << "Sending evenlop message_t: " << edef.type() << " seq " << edef.seq();

    // Create body message
    zmq::message_t zmqmsg(msg.ByteSizeLong());
    msg.SerializeToArray(zmqmsg.data(), zmqmsg.size());

    // Register callback first
    using ResponsePtr = std::unique_ptr<ResponseType>;
    Item args;
    if (done) {
        args = Item { ResponsePtr(new ResponseType), [done](const Status &s, ProtoPtr &&rep){
            done(s, ResponsePtr(static_cast<ResponseType*>(rep.release())));
        }};
    }
    {
        mutex_lock locker(m_mtable);
        m_recvCallbacks[seq] = std::move(args);
        return AsyncCallStarter(m_recvCallbacks[seq].typedCallbacks,
                                *this, seq, std::move(evenlop), std::move(zmqmsg));
    }
}

ZmqRpcClient::AsyncCallStarter ZmqRpcClient::rpcCallAsync(const ::google::protobuf::Message &msg)
{
    auto seq = m_seq.fetch_add(1);
    // Create evenlop message
    rpc::EvenlopDef edef;
    edef.set_type(msg.GetTypeName());
    edef.set_seq(seq);
    edef.set_recvidentity(m_recvId);
    edef.set_oplibrary(executor::TENSORFLOW);
    zmq::message_t evenlop(edef.ByteSizeLong());
    edef.SerializeToArray(evenlop.data(), evenlop.size());

    LOG(INFO) << "Sending evenlop message_t: " << edef.type() << " seq " << edef.seq();

    // Create body message
    zmq::message_t zmqmsg(msg.ByteSizeLong());
    msg.SerializeToArray(zmqmsg.data(), zmqmsg.size());

    {
        mutex_lock locker(m_mtable);
        m_recvCallbacks.emplace(seq, Item {});
        return AsyncCallStarter(m_recvCallbacks[seq].typedCallbacks,
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
Status ZmqRpcClient::rpcCall(const ::google::protobuf::Message &msg, std::unique_ptr<ResponseType> &reply)
{
    using ResponsePtr = std::unique_ptr<ResponseType>;

    Status status;
    Notification n;
    rpcCallAsync<ResponseType>(msg, [&n, &status, &reply](const Status &s, ResponsePtr &&rep){
        status = s;
        reply = std::move(rep);
        n.Notify();
    });

    n.WaitForNotification();

    return status;
}

void ZmqRpcClient::createSession(const ConfigProto & cfgProto,
                                 const FunctionDefLibrary & library, const GraphDef &graphdef)
{
    // FIXME: separate session creation on executor side
}

void ZmqRpcClient::runAsync(const ConfigProto &cfgProto, const FunctionDefLibrary &library, const Graph *graph,
                            AsyncOpKernel *kernel, OpKernelContext *context, AsyncOpKernel::DoneCallback done)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::runAsync";
    maybeInitialize(cfgProto, library, graph);

    rpc::RunRequest request;
    serializeOpKernel(request.mutable_opkernel(), kernel, graph, library, cfgProto);
    serializeOpContext(request.mutable_context(), context, graph, library, cfgProto);

    Item args;
    LOG(INFO) << "RpcClient::runAsync    calling rpc using rpc stub";
    using ResponsePtr = std::unique_ptr<rpc::RunResponse>;
    auto starter = rpcCallAsync<rpc::RunResponse>(request,
                                                  [done, context, this](const Status &status,
                                                                        ResponsePtr &&pResponse) {
        LOG(INFO) << "RpcClient::runAsync    rpc returned with status: "
                  << status.code() << " " << status.error_message();

        // TODO: better error handling
        if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
            return;
        }

        deserializeOpContext(context, &pResponse->context());
        done();
    });

    // NOTE: we cannot do RecvAsync inside the starter callback function, because RecvAsync may
    // call its cb on current thread, which in turn initiates another rpc call. So the starter
    // callback cannot block.
    Notification n;
    std::unique_ptr<executor::TFRendezRecvRequests> pReq;
    starter.add("executor.TFRendezRecvRequests", [&pReq, &n](const Status &s, ProtoPtr &&msg){
        pReq.reset(static_cast<executor::TFRendezRecvRequests*>(msg.release()));
        n.Notify();
    });
    starter.start();

    auto seq = starter.seq();
    n.WaitForNotification();
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
                                        [this, seq, parsed](const Status &s,
                                                        const Rendezvous::Args &send_args,
                                                        const Rendezvous::Args &recv_args,
                                                        const Tensor &val, bool is_dead){
            LOG(INFO) << "Send out executor.TFRendezRecvResponse for " << parsed.FullKey();
            rpc::TFRendezRecvResponse resp;
            resp.set_forseq(seq);
            auto item = resp.add_items();
            item->set_key(parsed.FullKey().ToString());
//             item->set_allocattributes(send_args.alloc_attrs.value);
            val.AsProtoTensorContent(item->mutable_val());

            rpc::CustomRequest request;
            request.set_type(resp.GetTypeName());
            resp.SerializeToString(request.mutable_extra());
            rpcCallAsync(request);
        });
    }
}

Status ZmqRpcClient::run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, const Graph *graph,
                         OpKernel *kernel, OpKernelContext *context)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::run";
    maybeInitialize(cfgProto, library, graph);

    rpc::RunRequest request;
    serializeOpKernel(request.mutable_opkernel(), kernel, graph, library, cfgProto);
    serializeOpContext(request.mutable_context(), context, graph, library, cfgProto);

    std::unique_ptr<rpc::RunResponse> pResponse;
    LOG(INFO) << "RpcClient::run    calling rpc using rpc stub";
    auto status = rpcCall(request, pResponse);
    LOG(INFO) << "RpcClient::run    rpc returned with status: "
              << status.code() << " " << status.error_message();

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    deserializeOpContext(context, &pResponse->context());

    return context->status();
}

Status ZmqRpcClient::allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle)
{
    LOG(INFO) << "RpcClient::allocate(alignment=" << alignment << ", num_bytes=" << num_bytes << ")";

    rpc::AllocRequest request;
    request.set_alignment(alignment);
    request.set_num_bytes(num_bytes);

    std::unique_ptr<rpc::AllocResponse> pResponse;
    auto status = rpcCall(request, pResponse);

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
    auto status = rpcCall(request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    return Status::OK();
}
} // namespace tensorflow
