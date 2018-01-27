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

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_rendezvous_mgr.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_wrapped_devicecontext.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <unordered_set>
#include <unordered_map>

namespace tensorflow {

bool ZrpcRemoteRendezvous::FindTensor(const std::string &key, Tensor &t)
{
    mutex_lock l(mu);
    auto it = tensors.find(key);
    if (it != tensors.end()) {
        t = it->second;
        return true;
    }
    return false;
}

Status ZrpcRemoteRendezvous::Send(const ParsedKey& key, const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead)
{
    // Must not hold lock when calling Send, which in turn may call cb, and requiring lock again
    {
        mutex_lock l(mu);
        tensors.emplace(key.FullKey().ToString(), val);
    }
    return BaseRemoteRendezvous::Send(key, args, val, is_dead);
}

void ZrpcRemoteRendezvous::RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                                     DoneCallback done)
{
    auto full_key = key.FullKey().ToString();
    auto final_done = [done, full_key, this](const Status &s, const Rendezvous::Args &send_args,
                                             const Rendezvous::Args &recv_args, const Tensor &val, bool is_dead){
        {
            mutex_lock l(mu);
            tensors.erase(full_key);
        }
        done(s, send_args, recv_args, val, is_dead);
    };

    return BaseRemoteRendezvous::RecvAsync(key, args, std::move(final_done));
}

void ZrpcRemoteRendezvous::SameWorkerRecvDone(const Rendezvous::ParsedKey &parsed,
                                              const Rendezvous::Args &send_args,
                                              const Rendezvous::Args &recv_args, const Tensor &in, Tensor *out,
                                              StatusCallback done)
{
    auto send_wrapper = static_cast<WrapperDeviceContext *>(send_args.device_context);
    auto recv_wrapper = static_cast<WrapperDeviceContext *>(recv_args.device_context);

    core::ScopedUnref send_unref(send_wrapper);
    core::ScopedUnref recv_unref(recv_wrapper);

    Device *send_dev = nullptr;
    DeviceContext *send_dctx = nullptr;
    if (send_wrapper) {
        send_dev = send_wrapper->device();
        send_dctx = send_wrapper->wrapped();
    } else {
        auto s = env_->device_mgr->LookupDevice(parsed.src_device, &send_dev);
        if (!s.ok()) {
            done(s);
            return;
        }
    }

    Device *recv_dev = nullptr;
    DeviceContext *recv_dctx = nullptr;
    if (recv_wrapper) {
        recv_dev = recv_wrapper->device();
        recv_dctx = recv_wrapper->wrapped();
    } else {
        auto s = env_->device_mgr->LookupDevice(parsed.dst_device, &recv_dev);
        if (!s.ok()) {
            done(s);
            return;
        }
    }

    // Do a quick copy (sharing the underlying buffer) if both tensors
    // are on host memory.
    const bool src_host =
    (send_args.alloc_attrs.on_host() || send_dev->attributes().device_type() == tensorflow::DEVICE_CPU);
    const bool dst_host =
    (recv_args.alloc_attrs.on_host() || recv_dev->attributes().device_type() == tensorflow::DEVICE_CPU);
    if (src_host && dst_host) {
        *out = in;
        done(Status::OK());
        return;
    }

    // This copy must involve a GPU. Hence, "in" must support DMA
    // (e.g., string tensors do not work on GPU).
    if (!DMAHelper::CanUseDMA(&in)) {
        done(errors::InvalidArgument("Non-DMA-safe ", DataTypeString(in.dtype()),
                                     " tensor may not be copied from/to a GPU."));
        return;
    }

    AllocatorAttributes attr = recv_args.alloc_attrs;
    attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                            recv_args.alloc_attrs.gpu_compatible());
    Allocator* out_allocator = recv_dev->GetAllocator(attr);
    Tensor copy(out_allocator, in.dtype(), in.shape());
    *out = copy;

    // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
    // etc.
    VLOG(1) << "ZrpcRemoteRendezvous::SameWorkerRecvDone copy from " << send_dev->name()
            << " to " << recv_dev->name() << "    send_on_host " << send_args.alloc_attrs.on_host()
            << " recv_on_host " << recv_args.alloc_attrs.on_host() << " src_data: " << (uint64_t)(in.tensor_data().data())
            << " dst_data: " << (uint64_t)(out->tensor_data().data());
    CopyTensor::ViaDMA(parsed.edge_name, send_dctx, recv_dctx, send_dev, recv_dev,
                        send_args.alloc_attrs, attr, &in, out,
                        done);
}

// Used only to retrieve tensors from remote processes.
class ZrpcRecvTensorCall : public BaseRecvTensorCall
{
public:
    ZrpcRecvTensorCall()
        : wi_(nullptr)
        , dst_device_(nullptr)
    {
    }

    void Init(WorkerInterface *wi, int64 step_id, StringPiece key, AllocatorAttributes alloc_attrs,
              Device *dst_device, const Rendezvous::Args &recv_args, Rendezvous::DoneCallback done)
    {
        wi_ = wi;
        alloc_attrs_ = alloc_attrs;
        dst_device_ = dst_device;
        recv_args_ = recv_args;
        done_ = std::move(done);
        req_.set_step_id(step_id);
        req_.set_rendezvous_key(key.data(), key.size());
    }

    void Reset(WorkerCacheInterface *wc)
    {
        wc->ReleaseWorker(src_worker_, wi_);
        wi_ = nullptr;
        alloc_attrs_ = AllocatorAttributes();
        dst_device_ = nullptr;
        // We don't clear opts_ and assume that Init will set up the state for
        // opts_ appropriately.
        req_.Clear();
        resp_.Clear();
        {
            mutex_lock l(mu_);
            status_ = Status::OK();
        }
        done_ = nullptr;
    }

    ~ZrpcRecvTensorCall() override
    {
        // Since only the ZrpcRecvTensorFreeList will delete an
        // ZrpcRecvTensorCall, and it always sets this->wi_ to null when
        // a call object is released to it, we can assert that this->wi_ is
        // always null at the point of deletion.
        CHECK_EQ(static_cast<WorkerInterface *>(nullptr), wi_)
            << "Leaking WorkerInterface in ZrpcRecvTensorCall destructor.";
    }

    void Start(std::function<void()> recv_done) override
    {
        StartRTCall(std::move(recv_done));
    }

    void StartAbort(const Status &s) override
    {
        {
            mutex_lock l(mu_);
            status_.Update(s);
        }
        opts_.StartCancel();
    }

    Status status() const override
    {
        mutex_lock l(mu_);
        return status_;
    }

    const Tensor &tensor() const
    {
        return resp_.tensor();
    }

    bool is_dead() const
    {
        return resp_.metadata().is_dead();
    }

    Device *dst_device() const
    {
        return dst_device_;
    }
    const Rendezvous::Args &recv_args() const
    {
        return recv_args_;
    }
    const Rendezvous::DoneCallback &done() const
    {
        return done_;
    }

private:
    friend class ZrpcRemoteRendezvous;

    // Start the main RecvTensor call, checking for an async abort.
    void StartRTCall(std::function<void()> recv_done)
    {
        resp_.InitAlloc(dst_device_, alloc_attrs_);
        using namespace std::placeholders;
        StatusCallback cb = std::bind(
            [this](std::function<void()> recv_done,
                   // Begin unbound arguments.
                   const Status &s) {
                if (!s.ok()) {
                    mutex_lock l(mu_);
                    status_.Update(s);
                }
                recv_done();
            },
            std::move(recv_done), _1);
        wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
    }

    string src_worker_;
    string src_rel_device_;
    WorkerInterface *wi_;
    AllocatorAttributes alloc_attrs_;
    Device *dst_device_;
    CallOptions opts_;
    RecvTensorRequest req_;
    TensorResponse resp_;
    Rendezvous::Args recv_args_;
    Rendezvous::DoneCallback done_;

    mutable mutex mu_;
    Status status_ GUARDED_BY(mu_);

    TF_DISALLOW_COPY_AND_ASSIGN(ZrpcRecvTensorCall);
};

class ZrpcRecvTensorFreeList
{
public:
    ZrpcRecvTensorFreeList()
    {
    }
    ~ZrpcRecvTensorFreeList()
    {
        for (size_t i = 0; i < objects_.size(); i++) {
            delete objects_[i];
        }
    }

    ZrpcRecvTensorCall *New()
    {
        {
            mutex_lock l(mu_);
            if (!objects_.empty()) {
                ZrpcRecvTensorCall *result = objects_.back();
                objects_.pop_back();
                return result;
            }
        }
        return new ZrpcRecvTensorCall;
    }

    void Release(ZrpcRecvTensorCall *obj, WorkerCacheInterface *wc)
    {
        obj->Reset(wc);
        {
            mutex_lock l(mu_);
            if (objects_.size() < kMaxObjects) {
                objects_.push_back(obj);
                return;
            }
        }
        delete obj;
    }

private:
    static const int kMaxObjects = 1000;

    mutex mu_;
    std::vector<ZrpcRecvTensorCall *> objects_ GUARDED_BY(mu_);
};

static ZrpcRecvTensorFreeList *get_call_freelist()
{
    static ZrpcRecvTensorFreeList *call_freelist = new ZrpcRecvTensorFreeList();
    return call_freelist;
}

// A private cache that wraps env->worker_cache and allows reuse of
// WorkerInterface objects.
class WorkerFreeListCache : public WorkerCacheInterface
{
public:
    explicit WorkerFreeListCache(WorkerCacheInterface *w)
        : wrapped_(w)
    {
    }

    ~WorkerFreeListCache()
    {
        for (auto p : workers_) {
            wrapped_->ReleaseWorker(p.first, p.second.worker);
        }
    }

    void ListWorkers(std::vector<string> *workers) const override
    {
        wrapped_->ListWorkers(workers);
    }

    WorkerInterface *CreateWorker(const string &target) override
    {
        mutex_lock l(mu_);
        auto p = workers_.find(target);
        if (p != workers_.end()) {
            return p->second.worker;
        }
        WorkerState state;
        state.worker = wrapped_->CreateWorker(target);
        if (state.worker != nullptr) {
            workers_.insert(std::make_pair(target, state));
        }
        return state.worker;
    }

    void ReleaseWorker(const string &target, WorkerInterface *worker) override
    {
        // TODO(jeff,sanjay): Should decrement ref-count when we implement eviction.
    }

    bool GetDeviceLocalityNonBlocking(const string &device, DeviceLocality *locality) override
    {
        return wrapped_->GetDeviceLocalityNonBlocking(device, locality);
    }

    void GetDeviceLocalityAsync(const string &device, DeviceLocality *locality, StatusCallback done) override
    {
        wrapped_->GetDeviceLocalityAsync(device, locality, done);
    }

    void SetLogging(bool active) override
    {
        wrapped_->SetLogging(active);
    }

    void ClearLogs() override
    {
        wrapped_->ClearLogs();
    }

    bool RetrieveLogs(int64 step_id, StepStats *ss) override
    {
        return wrapped_->RetrieveLogs(step_id, ss);
    }

private:
    WorkerCacheInterface *wrapped_;

    // Information kept per created WorkerInterface.
    struct WorkerState
    {
        WorkerInterface *worker;
        // TODO(jeff,sanjay): Add reference count if we support eviction.
    };

    // TODO(jeff,sanjay): Eviction when the map becomes too big.
    mutex mu_;
    std::unordered_map<string, WorkerState> workers_ GUARDED_BY(mu_);
};

void ZrpcRemoteRendezvous::RecvFromRemoteAsync(const Rendezvous::ParsedKey &parsed,
                                               const Rendezvous::Args &recv_args, DoneCallback done)
{
    CHECK(is_initialized());
    Status s;

    // Prepare a RecvTensor call that can handle being aborted.
    ZrpcRecvTensorCall *call = get_call_freelist()->New();

    // key.src_device identifies a remote device.
    if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_, &call->src_rel_device_)) {
        s = errors::Internal(parsed.src_device, " is invalid remote source device.");
    }
    WorkerSession *sess = session();
    WorkerInterface *rwi = sess->worker_cache->CreateWorker(call->src_worker_);
    if (s.ok() && rwi == nullptr) {
        s = errors::Internal("No worker known as ", call->src_worker_);
    }

    Device *dst_device;
    if (s.ok()) {
        s = sess->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
    }
    if (!s.ok()) {
        if (rwi != nullptr) {
            sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
        }
        get_call_freelist()->Release(call, sess->worker_cache.get());
        done(s, Args(), recv_args, Tensor{}, false);
        return;
    }

    call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device, recv_args,
               std::move(done));

    // Record "call" in active_ so that it can be aborted cleanly.
    RegisterCall(call);

    // Start "call".
    Ref();
    call->Start([this, call]() {
        // Removes "call" from active_. Prevent StartAbort().
        DeregisterCall(call);
        // If StartAbort was called prior to DeregisterCall, then the
        // current status should be bad.
        Status s = call->status();
        call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
        session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
        call->wi_ = nullptr;
        get_call_freelist()->Release(call, session()->worker_cache.get());
        Unref();
    });
}

ZrpcRendezvousMgr::ZrpcRendezvousMgr(const WorkerEnv *env)
    : BaseRendezvousMgr(env)
{
}

BaseRemoteRendezvous *ZrpcRendezvousMgr::Create(int64 step_id, const WorkerEnv *worker_env)
{
    return new ZrpcRemoteRendezvous(worker_env, step_id);
}

} // end namespace tensorflow
