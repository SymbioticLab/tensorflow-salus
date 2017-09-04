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
 */

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_ALLOCATORS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_ALLOCATORS_H_

#include "tensorflow/core/framework/allocator.h"

#include <memory>

namespace tensorflow {
namespace remote {
/**
 * @todo write docs
 */
class TrivialGPUAllocator : public Allocator
{
public:
    TrivialGPUAllocator(int gpu_id);

    ~TrivialGPUAllocator() override;

    // Return a string identifying this allocator
    std::string Name() override;

    // Return an uninitialized block of memory that is "num_bytes" bytes
    // in size.  The returned pointer is guaranteed to be aligned to a
    // multiple of "alignment" bytes.
    // REQUIRES: "alignment" is a power of 2.
    void *AllocateRaw(size_t alignment, size_t num_bytes) override;

    // Deallocate a block of memory pointer to by "ptr"
    // REQUIRES: "ptr" was previously returned by a call to AllocateRaw
    void DeallocateRaw(void *ptr) override;

private:
    string m_name;
    std::unique_ptr<SubAllocator> m_sub;
};

} // namespace remote
} // namespace tensorflow
#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_ALLOCATORS_H_
