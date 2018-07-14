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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DOUBLE_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DOUBLE_BFC_ALLOCATOR_H_

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/platform/mutex.h"

#include <memory>
#include <unordered_map>
#include <ostream>

namespace tensorflow {

// A GPU allocator that uses small allocation optimization to reduce fragmentation
class GPUDoubleBFCAllocator : public VisitableAllocator {
 public:
  TF_DISALLOW_COPY_AND_ASSIGN(GPUDoubleBFCAllocator);

  // 'device_id' refers to the StreamExecutor ID of the device within
  // the process and must reference a valid ID in the process.
  GPUDoubleBFCAllocator(int device_id, size_t total_memory);
  GPUDoubleBFCAllocator(int device_id, size_t total_memory,
                        const GPUOptions& gpu_options);
  ~GPUDoubleBFCAllocator() override {}

  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;

  void AddAllocVisitor(Visitor visitor) override;

  void AddFreeVisitor(Visitor visitor) override;

  bool TracksAllocationSizes() override { return true; }

  size_t RequestedSize(void* ptr) override;

  size_t AllocatedSize(void* ptr) override;

  int64 AllocationId(void* ptr) override;

  void GetStats(AllocatorStats* stats) override;

  void DumpMemoryLog() const;

  string GenerateMemoryMap() const;

 private:
  BFCAllocator *SelectAllocator(size_t num_bytes) const;

  std::ostream &GenerateMemoryMapForBFC(BFCAllocator* alloc, std::ostream &out) const;

  struct Chunk {
    BFCAllocator *allocator;
    int64 allocation_id;
  };
  Chunk FindAllocator(void* ptr) const;

  string name_;

  std::unique_ptr<BFCAllocator> small_alloc_;
  std::unique_ptr<BFCAllocator> big_alloc_;

  mutable mutex lock_;
  std::unordered_map<void*, Chunk> used_allocator GUARDED_BY(lock_);
  int64 next_allocation_id_ GUARDED_BY(lock_);
};

} // namespace tensorflow

#endif // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DOUBLE_BFC_ALLOCATOR_H_
