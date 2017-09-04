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

#include "tensorflow/core/distributed_runtime/zrpc/exechelper/allocators.h"

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace remote {

TrivialGPUAllocator::TrivialGPUAllocator(int gpu_id)
    : m_name(strings::StrCat("GPU_", gpu_id, "trivial"))
    , m_sub(new GPUMemAllocator(GPUMachineManager()->ExecutorForDevice(gpu_id).ValueOrDie()))
{
}

TrivialGPUAllocator::~TrivialGPUAllocator() = default;

std::string TrivialGPUAllocator::Name()
{
    return m_name;
}

void *TrivialGPUAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    return m_sub->Alloc(alignment, num_bytes);
}

void TrivialGPUAllocator::DeallocateRaw(void *ptr)
{
    return m_sub->Free(ptr);
}

} // namespace remote
} // namespace tensorflow
