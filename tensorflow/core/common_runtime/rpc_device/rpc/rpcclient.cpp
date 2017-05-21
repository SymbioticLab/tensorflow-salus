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

#include "rpcclient.h"

#include "grpc++/grpc++.h"

#include <sstream>

using namespace executor;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using std::shared_ptr;
using std::unique_ptr;
using std::ostringstream;

namespace tensorflow {

RpcClient::RpcClient()
    : RpcClient(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()))
{ }

RpcClient::RpcClient(shared_ptr<Channel> channel)
    : m_stub(IExecEngine::NewStub(channel))
{ }

RpcClient::~RpcClient() { }

Status RpcClient::run(const OpKernel *kernel, OpContext *context)
{
    // TODO: revisit and remove unnecessary copies of kernel and context.

    RunRequest request;
    *(request.mutable_opkernel()) = *kernel;
    request.set_allocated_context(context);

    RunResponse response;

    ClientContext grpc_context;

    auto status = m_stub->run(&grpc_context, request, &response);

    // TODO: better error handling
    if (status.ok() && response.result().code() == 0) {
        *context = response.context();
        return status;
    } else {
        ostringstream oss;
        oss << "ExecEngine returned " << response.result().code();
        return Status(grpc::StatusCode::ABORTED, oss.str());
    }
    return status;
}

AllocResponse RpcClient::allocate(uint64_t alignment, uint64_t num_bytes)
{
    AllocRequest request;
    return {};
}

DeallocResponse RpcClient::deallocate(uint64_t addr_handle)
{
    return {};
}

} // namespace tensorflow
