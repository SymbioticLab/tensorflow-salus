/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_worker_cache.h"

#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

namespace tensorflow {

namespace {

class ZrpcWorkerCache : public WorkerCachePartial
{
public:
    explicit ZrpcWorkerCache(WorkerInterface *local_worker, const string &local_target)
        : local_target_(local_target)
        , local_worker_(local_worker)
    {
    }

    // Explicit destructor to control destruction order.
    ~ZrpcWorkerCache() override
    {
    }

    void ListWorkers(std::vector<string> *workers) const override
    {
        workers->clear();
    }

    WorkerInterface *CreateWorker(const string &target) override
    {
        if (target == local_target_) {
            return local_worker_;
        } else {
            return nullptr;
        }
    }

    void ReleaseWorker(const string &target, WorkerInterface *worker) override
    {
        if (target == local_target_) {
            CHECK_EQ(worker, local_worker_) << "Releasing a worker that was not returned by this WorkerCache";
        } else {
            WorkerCachePartial::ReleaseWorker(target, worker);
        }
    }

    void SetLogging(bool v) override
    {
        logger_.SetLogging(v);
    }

    void ClearLogs() override
    {
        logger_.ClearLogs();
    }

    bool RetrieveLogs(int64 step_id, StepStats *ss) override
    {
        return logger_.RetrieveLogs(step_id, ss);
    }

private:
    const string local_target_;
    WorkerInterface *const local_worker_; // Not owned.
    WorkerCacheLogger logger_;
};

} // namespace

WorkerCacheInterface *NewZrpcWorkerCacheWithLocalWorker(WorkerInterface *local_worker,
                                                        const string &local_target)
{
    return new ZrpcWorkerCache(local_worker, local_target);
}

} // namespace tensorflow
