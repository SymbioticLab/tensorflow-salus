/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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

#include "tensorflow/core/common_runtime/gpu/gpu_double_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include <sstream>

namespace tensorflow {

static const size_t MAX_SMALL = 1 * 1024 * 1024; // 1MB
static const double SMALL_POOL = 500 * 1024 * 1024; // 200MB;

GPUDoubleBFCAllocator::GPUDoubleBFCAllocator(int device_id, size_t total_memory)
    : GPUDoubleBFCAllocator(device_id, total_memory, {}, true) {}

GPUDoubleBFCAllocator::GPUDoubleBFCAllocator(int device_id, size_t total_memory, const GPUOptions& gpu_options, bool use_small_opt)
    : name_(strings::StrCat("GPU_", device_id, "_dbfc"))
    , small_alloc_(nullptr)
    , big_alloc_(nullptr)
    , next_allocation_id_(1)
{
    if (use_small_opt) {
        CHECK(total_memory > SMALL_POOL) << "Total memory less than SMALL_POOL!!!";
        small_alloc_.reset(new BFCAllocator(
            new GPUMemAllocator(
                GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie()),
                SMALL_POOL, false,
            strings::StrCat("GPU_", device_id, "_bfc_small")));
        big_alloc_.reset(new BFCAllocator(
            new GPUMemAllocator(
                GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie()),
                total_memory - SMALL_POOL, gpu_options.allow_growth(),
            strings::StrCat("GPU_", device_id, "_bfc_big")));
    } else {
        big_alloc_.reset(new BFCAllocator(
            new GPUMemAllocator(
                GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie()),
                total_memory, gpu_options.allow_growth(),
            strings::StrCat("GPU_", device_id, "_bfc_big")));
    }
}

BFCAllocator *GPUDoubleBFCAllocator::SelectAllocator(size_t num_bytes) const
{
    if (small_alloc_ && num_bytes <= MAX_SMALL) {
        return small_alloc_.get();
    }
    return big_alloc_.get();
}

void* GPUDoubleBFCAllocator::AllocateRaw(size_t alignment, size_t num_bytes)
{
    auto alloc = SelectAllocator(num_bytes);
    auto ptr = alloc->AllocateRaw(alignment, num_bytes);

    if (ptr) {
        mutex_lock l(lock_);
        auto id = next_allocation_id_++;
        CHECK(used_allocator.emplace(ptr, Chunk{alloc, id}).second);
    }
    return ptr;
}

void* GPUDoubleBFCAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                         const AllocationAttributes& allocation_attr)
{
    auto alloc = SelectAllocator(num_bytes);
    auto ptr = alloc->AllocateRaw(alignment, num_bytes, allocation_attr);

    if (ptr) {
        mutex_lock l(lock_);
        auto id = next_allocation_id_++;
        CHECK(used_allocator.emplace(ptr, Chunk{alloc, id}).second);
    }
    return ptr;
}

GPUDoubleBFCAllocator::Chunk GPUDoubleBFCAllocator::FindAllocator(void* ptr) const
{
    if (!ptr) {
        return {};
    }
    mutex_lock l(lock_);
    auto it = used_allocator.find(ptr);
    if (it != used_allocator.end()) {
        return it->second;
    }
    return {};
}

void GPUDoubleBFCAllocator::DeallocateRaw(void* ptr)
{
    if (!ptr) {
        return;
    }

    BFCAllocator* alloc = nullptr;
    {
        mutex_lock l(lock_);
        auto it = used_allocator.find(ptr);
        if (it != used_allocator.end()) {
            alloc = it->second.allocator;
            used_allocator.erase(it);
        }
    }
    CHECK(alloc != nullptr)
        << "DeallocateRaw of pointer we never allocated: " << ptr;
    alloc->DeallocateRaw(ptr);
}

void GPUDoubleBFCAllocator::AddAllocVisitor(Visitor visitor)
{
    if (small_alloc_) {
        small_alloc_->AddAllocVisitor(std::move(visitor));
    }
    big_alloc_->AddAllocVisitor(std::move(visitor));
}

void GPUDoubleBFCAllocator::AddFreeVisitor(Visitor visitor)
{
    if (small_alloc_) {
        small_alloc_->AddFreeVisitor(std::move(visitor));
    }
    big_alloc_->AddFreeVisitor(std::move(visitor));
}

size_t GPUDoubleBFCAllocator::RequestedSize(void* ptr)
{
    auto alloc = FindAllocator(ptr).allocator;
    CHECK(alloc != nullptr)
        << "Asked for requested size of pointer we never allocated: " << ptr;
    return alloc->RequestedSize(ptr);
}

size_t GPUDoubleBFCAllocator::AllocatedSize(void* ptr)
{
    auto alloc = FindAllocator(ptr).allocator;
    CHECK(alloc != nullptr)
        << "Asked for allocated size of pointer we never allocated: " << ptr;
    return alloc->AllocatedSize(ptr);
}

int64 GPUDoubleBFCAllocator::AllocationId(void* ptr)
{
    auto id = FindAllocator(ptr).allocation_id;
    CHECK(id != 0)
        << "Asked for allocation id of pointer we never allocated: " << ptr;
    return id;
}

void GPUDoubleBFCAllocator::GetStats(AllocatorStats* stats)
{
    // based on big's stats
    big_alloc_->GetStats(stats);

    if (!small_alloc_) {
        return;
    }
    // add small
    AllocatorStats small;
    small_alloc_->GetStats(&small);

    stats->num_allocs += small.num_allocs;
    stats->bytes_in_use += small.bytes_in_use;
    stats->max_bytes_in_use += small.max_bytes_in_use;
    // use big's max_alloc_size
    // use big's bytes_limit
}

void GPUDoubleBFCAllocator::DumpMemoryLog() const
{
    {
        mutex_lock l(big_alloc_->lock_);
        big_alloc_->DumpMemoryLog(128);
    }
    if (small_alloc_) {
        mutex_lock l(small_alloc_->lock_);
        small_alloc_->DumpMemoryLog(128);
    }
}

std::ostream &GPUDoubleBFCAllocator::GenerateMemoryMapForBFC(BFCAllocator* alloc, std::ostream &out) const
{
    mutex_lock l(alloc->lock_);

    out << alloc->name_ << "\t";
    for (const auto& region : alloc->region_manager_.regions()) {
        auto h = alloc->region_manager_.get_handle(region.ptr());
        while (h != BFCAllocator::kInvalidChunkHandle) {
            const auto c = alloc->ChunkFromHandle(h);
            const size_t bsize = 0;
            if (c->bin_num != BFCAllocator::kInvalidBinNum) {
                bsize = alloc->BinFromIndex(c->bin_num)->bin_size;
            }
            out << c->ptr << "," << c->size << "," << c->in_use() << bsize << ";";
            h = c->next;
        }
    }
    out << "&";

    return out;
}

string GPUDoubleBFCAllocator::GenerateMemoryMap() const
{
    std::ostringstream oss;
    GenerateMemoryMapForBFC(big_alloc_.get(), oss);
    if (small_alloc_) {
        GenerateMemoryMapForBFC(small_alloc_.get(), oss);
    }
    return oss.str();
}

} // namespace tensorflow
