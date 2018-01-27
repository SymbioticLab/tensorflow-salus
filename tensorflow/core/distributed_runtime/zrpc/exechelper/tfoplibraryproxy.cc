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

#include "tfoplibraryproxy.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_util.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/zrpc/exechelper/mdgraphmgr.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace remote {

namespace {
const static char name_prefix[] = "/job:executor/replica:0/task:0";

thread::ThreadPool *computePool(Env *env)
{
    static std::unique_ptr<thread::ThreadPool> pool(new thread::ThreadPool(env, "ZrpcCompute", 4));
    return pool.get();
}

std::unique_ptr<Master> createMaster(MasterEnv *master_env)
{
    return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

} // namespace

TFOpLibraryProxy::TFOpLibraryProxy()
{
    m_env = Env::Default();
}

TFOpLibraryProxy::~TFOpLibraryProxy() = default;

Status TFOpLibraryProxy::globalInit(const ConfigProto &config)
{
    SessionOptions sess_opts;
    (*sess_opts.config.mutable_device_count())["RPC"] = 0;

    sess_opts.config.MergeFrom(config);

    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(sess_opts, name_prefix, &m_devices));
    m_deviceMgr.reset(new DeviceMgr(m_devices));
    return Status::OK();
}

Status TFOpLibraryProxy::newSession(std::unique_ptr<TFSessionProxy> &p)
{
    std::unique_ptr<TFSessionProxy> sess(new TFSessionProxy);
    auto s = sess->init(this);
    if (!s.ok()) {
        return s;
    }
    std::swap(sess, p);
    return Status::OK();
}

class TFSessionProxyPrivate
{
public:
    ~TFSessionProxyPrivate();

    MasterEnv masterEnv;
    std::unique_ptr<Master> master;
    WorkerEnv workerEnv;
    std::unique_ptr<ZrpcWorker> worker;

    ResourceMgr resourceMgr;
};

TFSessionProxyPrivate::~TFSessionProxyPrivate()
{
    delete masterEnv.worker_cache; // Shared with worker_env.worker_cache.

    // We must delete graph_mgr before device_mgr, due to shared
    // ownership of OpKernels in the executors. (The graph_mgr will
    // free all stateless OpKernels, and pass over borrowed stateful
    // OpKernels, which are also held in their respective devices'
    // OpSegments.)
    delete workerEnv.graph_mgr;

    delete workerEnv.rendezvous_mgr;

    // Do not delete (as these are not owned by the server):
    // - masterEnv.env
    // - workerEnv.env
    // - workerEnv.compute_pool
    // - workerEnv.device_mgr
}

TFSessionProxy::~TFSessionProxy() = default;

TFSessionProxy::TFSessionProxy()
    : d(new TFSessionProxyPrivate)
{
}

TFSessionProxy::TFSessionProxy(TFSessionProxy &&other)
    : d(std::move(other.d))
{
}

Status TFSessionProxy::init(TFOpLibraryProxy *proxy)
{
    d->masterEnv.env = proxy->m_env;
    d->workerEnv.env = proxy->m_env;

    // Configure shared devices between master and worker.
    for (auto dev : proxy->m_devices) {
        d->masterEnv.local_devices.push_back(dev);
    }
    d->workerEnv.device_mgr = proxy->m_deviceMgr.get();
    d->workerEnv.worker_name = name_prefix;

    // Create master and worker
    d->master = createMaster(&d->masterEnv);
    d->worker = NewZrpcWorker(&d->workerEnv);

    WorkerCacheInterface* worker_cache;
    WorkerCacheFactoryOptions worker_cache_factory_options;
    TF_RETURN_IF_ERROR(
        WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
    CHECK_NE(nullptr, worker_cache);

    // Finish setting up master environment.
    d->masterEnv.ops = OpRegistry::Global();
    d->masterEnv.worker_cache = worker_cache;
    d->masterEnv.master_session_factory =
        [config](SessionOptions options, const MasterEnv *env,
                 std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
                 std::unique_ptr<WorkerCacheInterface> worker_cache,
                 std::unique_ptr<DeviceSet> device_set) {
            options.config.MergeFrom(config);
            return new MasterSession(options, env, std::move(remote_devs),
                                     std::move(worker_cache), std::move(device_set),
                                     CreateNoOpStatsPublisher);
        };
    d->masterEnv.worker_cache_factory =
        [](const WorkerCacheFactoryOptions& options, WorkerCacheInterface** worker_cache) {
            return WorkerCacheFactory(options, worker_cache);
        };

    // Finish setting up worker environment.
    d->workerEnv.worker_cache = d->masterEnv.worker_cache;
    d->workerEnv.graph_mgr = new MDGraphMgr(&d->workerEnv);
    d->workerEnv.compute_pool = computePool(proxy->m_env);
    d->workerEnv.rendezvous_mgr = new ZrpcRendezvousMgr(&d->workerEnv);
    d->workerEnv.session_mgr = new SessionMgr(
        &d->workerEnv, "Salus",
        std::unique_ptr<WorkerCacheInterface>(worker_cache),
        [](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
            WorkerCacheFactoryOptions options(server_def);
            return WorkerCacheFactory(options, worker_cache);
        });

    return Status::OK();
}

void TFSessionProxy::setExecFactory(TFSessionProxy::ExecutorFactory f)
{
    static_cast<MDGraphMgr*>(d->workerEnv.graph_mgr)->setExecutorFactory(f);
}

void TFSessionProxy::schedule(std::function<void()> f)
{
    d->worker->env()->compute_pool->Schedule(std::move(f));
}

#define IMPL_MASTER_HANDLER(name)                                                                            \
    void TFSessionProxy::Handle##name(const name##Request *req,                                            \
                                      std::function<void(name##Response *, Status)> cb)                    \
    {                                                                                                        \
        auto resp = new name##Response();                                                                    \
        d->master->name(req, resp, [cb, resp](const Status &s) { cb(resp, s); });                          \
    }

IMPL_MASTER_HANDLER(ExtendSession)
IMPL_MASTER_HANDLER(PartialRunSetup)
IMPL_MASTER_HANDLER(CloseSession)
IMPL_MASTER_HANDLER(ListDevices)
IMPL_MASTER_HANDLER(Reset)

#undef IMPL_MASTER_HANDLER

void TFSessionProxy::HandleCreateSession(const CreateSessionRequest *req,
                                   std::function<void(CreateSessionResponse *, Status)> cb)
{
    auto resp = new CreateSessionResponse();
    d->master->CreateSession(req, resp, [cb, resp](const Status &s) { cb(resp, s); });
}

void TFSessionProxy::HandleRunStep(const RunStepRequest *req,
                                   std::function<void(RunStepResponse *, Status)> cb)
{
    CallOptions *call_opts = new CallOptions;
    auto wrapped_request = new ProtoRunStepRequest(req);
    auto resp = new RunStepResponse();
    auto wrapped_response = new NonOwnedProtoRunStepResponse(resp);
    d->master->RunStep(call_opts, wrapped_request, wrapped_response,
                       [call_opts, wrapped_request, wrapped_response, resp, cb](const Status &status) {
                           delete call_opts;
                           delete wrapped_request;
                           delete wrapped_response;
                           cb(resp, status);
                       });
}

#define IMPL_WORKER_HANDLER(name)                                                                            \
    void TFSessionProxy::Handle##name(const name##Request *req,                                            \
                                      std::function<void(name##Response *, Status)> cb)                    \
    {                                                                                                        \
        schedule([this, req, cb]() {                                                                      \
            auto resp = new name##Response();                                                                \
            d->worker->name##Async(req, resp, [resp, cb](Status s) { cb(resp, s); });                      \
        });                                                                                                  \
    }

CallWithWorkerMethodName(IMPL_WORKER_HANDLER)

#undef IMPL_WORKER_HANDLER

void TFSessionProxy::HandleRunGraph(const RunGraphRequest *req,
                                    std::function<void(RunGraphResponse *, Status)> cb)
{
    schedule([this, req, cb]() {
        CallOptions *call_opts = new CallOptions;
        auto wrapped_request = new ProtoRunGraphRequest(req);
        auto resp = new RunGraphResponse();
        auto wrapped_response = new NonOwnedProtoRunGraphResponse(resp);
        d->worker->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                                 [call_opts, wrapped_request, wrapped_response, resp, cb](const Status &s) {
                                     delete call_opts;
                                     delete wrapped_request;
                                     delete wrapped_response;
                                     cb(resp, s);
                                 });
    });
}

void TFSessionProxy::HandleRecvTensorRaw(const RecvTensorRequest *req,
                                         std::function<void(std::vector<zmq::message_t> *, Status)> cb)
{
    schedule([this, req, cb]() {
        CallOptions *call_opts = new CallOptions;
        auto resp = new zmq::MultiPartMessage;
        d->worker->RecvTensorAsync(call_opts, req, resp, [call_opts, resp, cb](const Status &s) {
            delete call_opts;
            cb(resp->release(), s);
        });
    });
}

} // namespace remote
} // namespace tensorflow
