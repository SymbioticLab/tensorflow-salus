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

#include "tensorflow/core/lib/core/status.h"

#include <memory>

namespace executor {
class OpKernelDef;
class OpContextDef;
}

namespace tensorflow {

class OpKernel;
class OpKernelContext;

/**
 * @todo write docs
 */
class RpcClient
{
public:
    RpcClient();

    virtual ~RpcClient();

    virtual Status run(OpKernel *kernel, OpKernelContext *context) = 0;
    virtual Status allocate(uint64_t alignment, uint64_t num_bytes, uint64_t *addr_handle) = 0;
    virtual Status deallocate(uint64_t addr_handle) = 0;

    // default instance always connect to localhost:5501
    static RpcClient &instance();

    void serializeOpKernel(executor::OpKernelDef *def, const OpKernel *kernel);
    void serializeOpContext(executor::OpContextDef *def, const OpKernelContext *context);
    void deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def);

private:
};

} // namespace tensorflow

#endif // RPCCLIENT_H
