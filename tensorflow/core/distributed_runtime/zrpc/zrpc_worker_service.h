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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_WORKER_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_WORKER_SERVICE_H_

#include "tensorflow/core/distributed_runtime/worker.h"

#include <memory>

namespace zmq {
class MultiPartMessage;
}

namespace tensorflow {

struct WorkerEnv;

class ZrpcWorker : public Worker
{
public:
    ZrpcWorker(WorkerEnv *env);

    // Specialized version of RecvTensor for ZeroMQ, which avoids a copy.
    void RecvTensorAsync(CallOptions *opts, const RecvTensorRequest *request, zmq::MultiPartMessage *response,
                         StatusCallback done);

    WorkerEnv *env();
};

std::unique_ptr<ZrpcWorker> NewZrpcWorker(WorkerEnv *worker_env);

} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_ZRPC_WORKER_SERVICE_H_
