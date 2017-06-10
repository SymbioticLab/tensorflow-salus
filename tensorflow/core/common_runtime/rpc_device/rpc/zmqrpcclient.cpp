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

#include <sstream>
#include <cstring>

namespace rpc = ::executor;
using std::shared_ptr;
using std::unique_ptr;
using std::ostringstream;

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

ZmqRpcClient::ZmqRpcClient()
    : m_zmqctx(1)
    , m_zmqsock(m_zmqctx, ZMQ_REQ)
{
    LOG(INFO) << "Created ZeroMQ context";
    try {
        m_zmqsock.connect("tcp://localhost:5501");
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "ZeroMQ socket connect failed: " << err.what();
    }
}

ZmqRpcClient::~ZmqRpcClient() { }

Status ZmqRpcClient::rpcCall(::google::protobuf::Message &msg, ::google::protobuf::Message &reply)
{
    try {
        // Create evenlop message
        auto type = msg.GetTypeName();
        zmq::message_t evenlop(type.size());
        memcpy(evenlop.data(), type.c_str(), evenlop.size());

        LOG(INFO) << "Sending evenlop message_t: " << type;

        // Create body message
        zmq::message_t zmqmsg(msg.ByteSizeLong());
        // TODO: consider remove the copy
        msg.SerializeToArray(zmqmsg.data(), zmqmsg.size());

        zmq::message_t rbody;
        {
            mutex_lock locker(m_mu);
            LOG(INFO) << "Sending body message_t of size: " << zmqmsg.size();
            m_zmqsock.send(evenlop, ZMQ_SNDMORE);
            m_zmqsock.send(zmqmsg);

            LOG(INFO) << "Sending Returned, waiting for reply";

            // Receive reply
            m_zmqsock.recv(&rbody);
            LOG(INFO) << "Got reply of size: " << rbody.size();
        }

        // Parse reply
        if(reply.ParseFromArray(rbody.data(), rbody.size())) {
            return Status::OK();
        } else {
            return Status(error::INTERNAL, "Malformated message");
        }
    } catch (zmq::error_t &err) {
        LOG(ERROR) << "ZeroMQ socket connect failed: " << err.what();
        return Status(error::INTERNAL, err.what());
    }
}

Status ZmqRpcClient::run(const ConfigProto &cfgProto, const FunctionDefLibrary &library, Graph *graph,
                         OpKernel *kernel, OpKernelContext *context)
{
    LOG(INFO) << "===================================================================";
    LOG(INFO) << "RpcClient::run";

    rpc::RunRequest request;
    serializeOpKernel(request.mutable_opkernel(), kernel, graph, library, cfgProto);
    serializeOpContext(request.mutable_context(), context, graph, library, cfgProto);

    rpc::RunResponse response;
    LOG(INFO) << "RpcClient::run    calling rpc using rpc stub";
    auto status = rpcCall(request, response);
    LOG(INFO) << "RpcClient::run    rpc returned with status: "
              << status.code() << " " << status.error_message();

    // TODO: better error handling
    if (!status.ok() || response.result().code() != 0) {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(error::ABORTED, oss.str());
    }

    deserializeOpContext(context, &response.context());

    return context->status();
}

Status ZmqRpcClient::allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle)
{
    LOG(INFO) << "RpcClient::allocate(alignment=" << alignment << ", num_bytes=" << num_bytes << ")";

    rpc::AllocRequest request;
    request.set_alignment(alignment);
    request.set_num_bytes(num_bytes);

    rpc::AllocResponse response;
    auto status = rpcCall(request, response);

    // TODO: better error handling
    if (!status.ok() || response.result().code() != 0) {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(error::ABORTED, oss.str());
    }

    *addr_handle = response.addr_handle();
    LOG(INFO) << "RpcClient::allocate returned addr_handle=" << addr_handle;
    return Status::OK();
}

Status ZmqRpcClient::deallocate(uint64_t addr_handle)
{
    LOG(INFO) << "RpcClient::deallocate(addr_handle=" << addr_handle;

    rpc::DeallocRequest request;
    request.set_addr_handle(addr_handle);

    rpc::DeallocResponse response;
    auto status = rpcCall(request, response);

    // TODO: better error handling
    if (!status.ok() || response.result().code() != 0) {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(error::ABORTED, oss.str());
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
    rpc::FetchResponse response;
    auto status = rpcCall(request, response);

    // TODO: better error handling
    if (!status.ok() || response.result().code() != 0) {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(error::ABORTED, oss.str());
    }

    LOG(INFO) << "Got fetch response: " << response.DebugString();

    rpc::TFTensors recved;
    recved.ParseFromString(response.extra());

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
    rpc::PushResponse response;
    auto status = rpcCall(request, response);

    // TODO: better error handling
    if (!status.ok() || response.result().code() != 0) {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(error::ABORTED, oss.str());
    }

    return Status::OK();
}

} // namespace tensorflow
