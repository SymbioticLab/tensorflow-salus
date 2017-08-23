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

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_worker_service.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_tensor_coding.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

#include "zmq.hpp"

#include <deque>

namespace tensorflow {

ZrpcWorker::ZrpcWorker(WorkerEnv *worker_env)
    : Worker(worker_env)
{
}

// RecvTensorAsync: unlike the other Worker methods, which use protocol buffers
// for a response object, to avoid extra protocol buffer serialization overhead
// we generate our response directly into a ::zmq::message_t object
void ZrpcWorker::RecvTensorAsync(CallOptions *opts, const RecvTensorRequest *request,
                                 zmq::MultiPartMessage *response, StatusCallback done)
{
    const int64 step_id = request->step_id();
    const string &key = request->rendezvous_key();
    TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
    Rendezvous::ParsedKey parsed;
    Status s = Rendezvous::ParseKey(key, &parsed);
    Device *src_dev = nullptr;
    if (s.ok()) {
        s = PrepareRecvTensor(parsed, &src_dev);
    }
    if (!s.ok()) {
        done(s);
        return;
    }

    // Request the tensor associated with the rendezvous key. Any time
    // while waiting for the tensor to be produced, up until the start
    // of execution of the callback lambda body below, an RPC
    // cancellation should abort the rendezvous.
    opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
    env_->rendezvous_mgr->RecvLocalAsync(step_id, parsed, [opts, response, done,
                                                           src_dev](const Status &status,
                                                                    const Rendezvous::Args &send_args,
                                                                    const Rendezvous::Args &recv_args,
                                                                    const Tensor &val, const bool is_dead) {
        opts->ClearCancelCallback();
        if (status.ok()) {
            // DMA can only be used for Tensors that do not fall into
            // the following three odd edge cases: 1) a zero-size
            // buffer, 2) a dead tensor which has an uninit value, and
            // 3) the tensor has the on_host allocation attribute,
            // i.e. it's in CPU RAM *independent of its assigned
            // device type*.
            const bool on_host = send_args.alloc_attrs.on_host();
            {
                // Non-DMA cases.
                if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
#if GOOGLE_CUDA
                    const DeviceContext *send_dev_context = send_args.device_context;
                    RecvTensorResponse *tmp = new RecvTensorResponse;
                    tmp->set_is_dead(is_dead);
                    CHECK(send_dev_context) << "send dev name: " << src_dev->name()
                                            << " gpu_info: " << src_dev->tensorflow_gpu_device_info();
                    // "val" is on a GPU. Uses GPUUtil to fill the response proto.
                    StatusCallback response_ready = [response, done, tmp](const Status &s) {
                        // The value is now ready to be returned on the wire.
                        tmp->set_send_start_micros(Env::Default()->NowMicros());

                        EncodeRecvTensorResponseToByteBuffer(*tmp, *response);
                        done(s);
                        delete tmp;
                    };

                    // TODO (jeff,sanjay,mrry): Avoid copy on GPU path by
                    // modifying GPUUtil::SetProtoFromGPU to accept a
                    // ::grpc::ByteBuffer to serialize to, rather than
                    // encoding into a protocol buffer and then
                    // serializing that (i.e. figure out how to use
                    // EncodeTensorToByteBuffer on this path rather than
                    // EncodeRecvTensorResponseToByteBuffer)
                    GPUUtil::SetProtoFromGPU(val, src_dev, send_dev_context, tmp->mutable_tensor(), is_dead,
                                             response_ready);
#else
              done(errors::Internal("No GPU device in process"));
#endif // GOOGLE_CUDA
                } else {
                    EncodeTensorToByteBuffer(is_dead, val, *response);
                    done(Status::OK());
                }
            }
        } else {
            //  !s.ok()
            done(status);
        }
    });
}

WorkerEnv *ZrpcWorker::env()
{
    return env_;
}

std::unique_ptr<ZrpcWorker> NewZrpcWorker(WorkerEnv *env)
{
    return std::unique_ptr<ZrpcWorker>(new ZrpcWorker(env));
}

} // namespace tensorflow
