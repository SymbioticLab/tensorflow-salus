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

#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.grpc.pb.h"

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
    /**
     * Default constructor. Hardcoded to connect to localhost:50051
     */
    RpcClient();


    ~RpcClient();

    grpc::Status run(OpKernel *kernel, OpKernelContext *context);
    grpc::Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle);
    grpc::Status deallocate(uint64_t addr_handle);

    static RpcClient &instance();

private:
    RpcClient(std::shared_ptr<grpc::Channel> channel);

    std::unique_ptr<executor::IExecEngine::Stub> m_stub;
};

} // namespace tensorflow

#endif // RPCCLIENT_H
