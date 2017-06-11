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

namespace {
void tensorToProto(tensorflow::TensorProto *proto, const tensorflow::Tensor &tensor)
{
    proto->set_dtype(tensor.dtype());
    tensor.shape().AsProto(proto->mutable_tensor_shape());

    auto addr_handle = reinterpret_cast<uint64_t>(tensor.tensor_data().data());
    // HACK: use a int64 val entry to store the addr handle for simplicity,
    // idealy should store this in tensor_content with proper encoding.
    proto->add_int64_val(addr_handle);
}

}

namespace tensorflow {

ZmqRpcClient::ZmqRpcClient(Env *env, const std::string &executorAddr)
    : m_execAddr(executorAddr)
    , m_zmqctx(1)
    , m_seq(0)
    , m_sendSock(m_zmqctx, ZMQ_REQ)
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
    // close context first, so that recv thread will return.
    m_zmqctx.close();

    delete m_recvThread;
}

ZmqRpcClient::Item::Item() {}

ZmqRpcClient::Item::Item(Item &&other) : reply(std::move(other.reply)) , done(std::move(other.done)) {}

ZmqRpcClient::Item::Item(ProtoPtr &&rep, RawDoneCallback done) : reply(std::move(rep)), done(done) {}

ZmqRpcClient::Item &ZmqRpcClient::Item::operator=(Item &&other)
{
    std::swap(this->reply, other.reply);
    std::swap(this->done, other.done);
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

            LOG(INFO) << "Received evenlop: seq=" << edef.seq();

            // Find corresonding item in table
            Item item;
            {
                mutex_lock locker(m_mtable);
                auto it = m_recvCallbacks.find(edef.seq());
                if (it == m_recvCallbacks.end()) {
                    LOG(ERROR) << "Skipped one iteration due to seq not found in table: " << edef.seq();
                    continue;
                }
                std::swap(item ,it->second);
                m_recvCallbacks.erase(it);
            }

            if (!item.done) {
                LOG(WARNING) << "Skipped one iteration due to no callback";
                continue;
            }

            // Now receive our message body
            if (!recvSock.getsockopt<int64_t>(ZMQ_RCVMORE)) {
                LOG(ERROR) << "Skipped one iteration due to no body message part found after identity frames";
                item.done(errors::Internal("No body message found"), std::move(item.reply));
                continue;
            }
            recvSock.recv(&msg);
            if(!item.reply->ParseFromArray(msg.data(), msg.size())) {
                LOG(ERROR) << "Received malformatted message body. Dropping";
                item.done(errors::Internal("Body message malformatted"), std::move(item.reply));
                continue;
            }

            item.done(Status::OK(), std::move(item.reply));
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
void ZmqRpcClient::rpcCallAsync(const google::protobuf::Message& msg,
                                std::function<void(const Status&, std::unique_ptr<ResponseType>&&)> done)
{
    auto seq = m_seq.fetch_add(1);
    try {
        // Create evenlop message
        rpc::EvenlopDef edef;
        edef.set_type(msg.GetTypeName());
        edef.set_seq(seq);
        edef.set_recvidentity(m_recvId);
        zmq::message_t evenlop(edef.ByteSizeLong());
        edef.SerializeToArray(evenlop.data(), evenlop.size());

        LOG(INFO) << "Sending evenlop message_t: " << edef.type() << " seq " << edef.seq();

        // Create body message
        zmq::message_t zmqmsg(msg.ByteSizeLong());
        msg.SerializeToArray(zmqmsg.data(), zmqmsg.size());

        // Register callback first
        using ResponsePtr = std::unique_ptr<ResponseType>;
        Item args { ResponsePtr(new ResponseType), [done](const Status &s, ProtoPtr &&rep){
            done(s, ResponsePtr(static_cast<ResponseType*>(rep.release())));
        }};
        {
            mutex_lock locker(m_mtable);
            m_recvCallbacks[seq] = std::move(args);
        }

        // Send out message
        {
            mutex_lock locker(m_mu);
            m_sendSock.send(evenlop, ZMQ_SNDMORE);
            m_sendSock.send(zmqmsg);
            LOG(INFO) << "Sent body message_t of size: " << zmqmsg.size();
        }
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "Error when sending message seq " << seq << ": " << err.what();
        // cleanup callback if any
        mutex_lock locker(m_mtable);
        m_recvCallbacks.erase(seq);
    }
}

template<typename ResponseType>
Status ZmqRpcClient::rpcCall(const ::google::protobuf::Message &msg, std::unique_ptr<ResponseType> &reply)
{
    using ResponsePtr = std::unique_ptr<ResponseType>;

    Status status;
    Notification n;
    Item args;
    rpcCallAsync<ResponseType>(msg, [&n, &status, &reply](const Status &s, ResponsePtr &&rep){
        status = s;
        reply = std::move(rep);
        n.Notify();
    });

    n.WaitForNotification();

    return status;
}

void ZmqRpcClient::createSession(const ConfigProto & cfgProto,
                                 const FunctionDefLibrary & library, Graph *graph)
{
    // FIXME: separate session creation on executor side
}

void ZmqRpcClient::runAsync(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
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
    rpcCallAsync<rpc::RunResponse>(request,
                                   [done, context, this](const Status &status, ResponsePtr &&pResponse) {
        LOG(INFO) << "RpcClient::runAsync    rpc returned with status: "
                  << status.code() << " " << status.error_message();

        // TODO: better error handling
        if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
            return;
        }

        deserializeOpContext(context, &pResponse->context());
        done();
    });
}

Status ZmqRpcClient::run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
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

Status ZmqRpcClient::fetch(tensorflow::Tensor *cpu_tensor, const tensorflow::Tensor *dev_tensor)
{
    LOG(INFO) << "RpcClient::fetch";

    rpc::TFTensors tensors;
    tensorToProto(tensors.add_tensors(), *dev_tensor);

    rpc::FetchRequest request;
    request.set_oplibrary(rpc::TENSORFLOW);
    tensors.SerializeToString(request.mutable_extra());

    LOG(INFO) << "RpcCLient::fetch actual request: " << request.DebugString();

    // Actuall call
    std::unique_ptr<rpc::FetchResponse> pResponse;
    auto status = rpcCall(request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    LOG(INFO) << "Got fetch response: " << pResponse->DebugString();

    rpc::TFTensors recved;
    recved.ParseFromString(pResponse->extra());

    LOG(INFO) << "Got parsed tftensors: " << recved.DebugString();

    if (recved.tensors_size() != 1) {
        LOG(ERROR) << "Parsed proto contains wrong number of tensor";
        return errors::Internal("Failed to parse proto");
    }

    auto recvedproto = recved.tensors(0);

    if (!cpu_tensor->FromProto(recvedproto)) {
        LOG(ERROR) << "Failed to parse proto: " << recvedproto.DebugString();
        return errors::Internal("Failed to parse proto");
    }

    return Status::OK();
}

Status ZmqRpcClient::push(tensorflow::Tensor *dev_tensor, const tensorflow::Tensor *cpu_tensor)
{
    LOG(INFO) << "RpcClient::push";

    rpc::TFPushRequest push;
    cpu_tensor->AsProtoTensorContent(push.add_data());
    tensorToProto(push.add_tensors(), *dev_tensor);

    rpc::PushRequest request;
    request.set_oplibrary(rpc::TENSORFLOW);
    push.SerializeToString(request.mutable_extra());

    // Actuall call
    std::unique_ptr<rpc::PushResponse> pResponse;
    auto status = rpcCall(request, pResponse);

    // TODO: better error handling
    if (!status.ok() || !pResponse || pResponse->result().code() != 0) {
        return status;
    }

    return Status::OK();
}

} // namespace tensorflow
