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

#include "rpc_allocator.h"

#include "rpc/rpcclient.h"

#include <memory>

namespace tensorflow {

RPCAllocator::RPCAllocator(RpcClient &rpc)
    : m_rpc(rpc)
{

}

RPCAllocator::~RPCAllocator()
{

}

string RPCAllocator::Name()
{
    return "rpc";
}

void *RPCAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    uint64_t addr_handle;
    LOG(INFO) << "RpcAllocateRaw with alignment " << alignment << " num_bytes " << num_bytes;

    auto status = m_rpc.allocate(alignment, num_bytes, &addr_handle);
    if (status.ok()) {
        return reinterpret_cast<void*>(addr_handle);
    }
    return nullptr;
}

void RPCAllocator::DeallocateRaw(void *ptr)
{
    LOG(INFO) << "RpcDeallocateRaw with ptr " << reinterpret_cast<uint64_t>(ptr);
    auto status = m_rpc.deallocate(reinterpret_cast<uint64_t>(ptr));
    if (!status.ok()) {
        LOG(ERROR) << "Error in RpcAllocator::DeallocateRaw";
    }
}

std::unique_ptr<OneTimeAllocator> OneTimeAllocator::create(uint64_t addr)
{
    return std::unique_ptr<OneTimeAllocator>(new OneTimeAllocator(addr));
}

OneTimeAllocator::OneTimeAllocator(uint64_t addr_handle)
    : m_addr_handle(addr_handle)
{ }

OneTimeAllocator::~OneTimeAllocator() = default;

string OneTimeAllocator::Name()
{
    return "onetime";
}

void *OneTimeAllocator::AllocateRaw(size_t /* alignment */, size_t /* num_bytes */)
{
    return reinterpret_cast<void*>(m_addr_handle);
}

void OneTimeAllocator::DeallocateRaw(void *ptr)
{
    CHECK(reinterpret_cast<uint64_t>(ptr) == m_addr_handle);
    delete this;
}

} // namespace tensorflow
