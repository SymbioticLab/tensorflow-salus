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

class TFOpLibraryProxyPrivate
{
public:
    TFOpLibraryProxyPrivate(TFOpLibraryProxy *q)
        : q(q)
    {
    }

    ~TFOpLibraryProxyPrivate()
    {
        delete m_masterEnv.worker_cache; // Shared with worker_env.worker_cache.

        // We must delete graph_mgr before device_mgr, due to shared
        // ownership of OpKernels in the executors. (The graph_mgr will
        // free all stateless OpKernels, and pass over borrowed stateful
        // OpKernels, which are also held in their respective devices'
        // OpSegments.)
        delete m_workerEnv.graph_mgr;
        delete m_workerEnv.device_mgr;

        delete m_workerEnv.rendezvous_mgr;

        // Do not delete (as these are not owned by the server):
        // - m_masterEnv.env
        // - m_workerEnv.env
        // - m_workerEnv.compute_pool
    }

    std::unique_ptr<Master> CreateMaster(MasterEnv *master_env)
    {
        return std::unique_ptr<Master>(new Master(master_env, 0.0));
    }

    Status init()
    {
        mutex_lock l(m_mu);
        m_masterEnv.env = m_env;
        m_workerEnv.env = m_env;

        SessionOptions sess_opts;
        (*sess_opts.config.mutable_device_count())["RPC"] = 0;

        // Configure shared devices between master and worker.
        string name_prefix = strings::StrCat("/job:", "executor", "/replica:0", "/task:", 0);
        TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(sess_opts, name_prefix, &m_masterEnv.local_devices));
        m_workerEnv.device_mgr = new DeviceMgr(m_masterEnv.local_devices);
        m_workerEnv.worker_name = name_prefix;

        // Create master and worker
        m_master = CreateMaster(&m_masterEnv);
        m_worker = NewZrpcWorker(&m_workerEnv);

        // Finish setting up master environment.
        m_masterEnv.ops = OpRegistry::Global();
        m_masterEnv.worker_cache = NewZrpcWorkerCacheWithLocalWorker(m_worker.get(), name_prefix);
        m_masterEnv.master_session_factory =
            [](const SessionOptions &options, const MasterEnv *env,
               std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs) {
                return new MasterSession(options, env, std::move(remote_devs), CreateNoOpStatsPublisher);
            };

        // Finish setting up worker environment.
        m_workerEnv.worker_cache = m_masterEnv.worker_cache;
        m_workerEnv.graph_mgr = new MDGraphMgr(&m_workerEnv, m_execFactory);
        // TODO: use our own thread pool
        m_workerEnv.compute_pool = ComputePool(sess_opts);
        m_workerEnv.rendezvous_mgr = new ZrpcRendezvousMgr(&m_workerEnv);

        return Status::OK();
    }

    void schedule(std::function<void()> f)
    {
        m_worker->env()->compute_pool->Schedule(std::move(f));
    }

    mutex m_mu;

    Env *m_env;
    MasterEnv m_masterEnv;
    std::unique_ptr<Master> m_master;
    WorkerEnv m_workerEnv;
    std::unique_ptr<ZrpcWorker> m_worker;
    TFOpLibraryProxy::ExecutorFactory m_execFactory;

private:
    tensorflow::remote::TFOpLibraryProxy *const q;
};

TFOpLibraryProxy::~TFOpLibraryProxy() = default;

TFOpLibraryProxy::TFOpLibraryProxy(ExecutorFactory execFactory)
    : d(new TFOpLibraryProxyPrivate(this))
{
    d->m_env = Env::Default();
    d->m_execFactory = execFactory;
}

TFOpLibraryProxy::TFOpLibraryProxy(TFOpLibraryProxy &&other)
    : d(std::move(other.d))
{
}

Status TFOpLibraryProxy::init()
{
    return d->init();
}

#define IMPL_MASTER_HANDLER(name)                                                                            \
    void TFOpLibraryProxy::Handle##name(const name##Request *req,                                            \
                                        std::function<void(name##Response *, Status)> cb)                    \
    {                                                                                                        \
        auto resp = new name##Response();                                                                    \
        d->m_master->name(req, resp, [cb, resp](const Status &s) { cb(resp, s); });                          \
    }

CallWithMasterMethodName(IMPL_MASTER_HANDLER)

#undef IMPL_MASTER_HANDLER

void TFOpLibraryProxy::HandleRunStep(const RunStepRequest *req,
                                     std::function<void(RunStepResponse *, Status)> cb)
{
    CallOptions *call_opts = new CallOptions;
    auto wrapped_request = new ProtoRunStepRequest(req);
    auto resp = new RunStepResponse();
    auto wrapped_response = new NonOwnedProtoRunStepResponse(resp);
    d->m_master->RunStep(call_opts, wrapped_request, wrapped_response,
                         [call_opts, wrapped_request, wrapped_response, resp, cb](const Status &status) {
                             delete call_opts;
                             delete wrapped_request;
                             delete wrapped_response;
                             cb(resp, status);
                         });
}

#define IMPL_WORKER_HANDLER(name)                                                                            \
    void TFOpLibraryProxy::Handle##name(const name##Request *req,                                            \
                                        std::function<void(name##Response *, Status)> cb)                    \
    {                                                                                                        \
        d->schedule([this, req, cb]() {                                                                      \
            auto resp = new name##Response();                                                                \
            d->m_worker->name##Async(req, resp, [resp, cb](Status s) { cb(resp, s); });                      \
        });                                                                                                  \
    }

CallWithWorkerMethodName(IMPL_WORKER_HANDLER)

#undef IMPL_WORKER_HANDLER

void TFOpLibraryProxy::HandleRunGraph(const RunGraphRequest *req,
                                      std::function<void(RunGraphResponse *, Status)> cb)
{
    d->schedule([this, req, cb]() {
        CallOptions *call_opts = new CallOptions;
        auto wrapped_request = new ProtoRunGraphRequest(req);
        auto resp = new RunGraphResponse();
        auto wrapped_response = new NonOwnedProtoRunGraphResponse(resp);
        d->m_worker->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                                   [call_opts, wrapped_request, wrapped_response, resp, cb](const Status &s) {
                                       delete call_opts;
                                       delete wrapped_request;
                                       delete wrapped_response;
                                       cb(resp, s);
                                   });
    });
}

void TFOpLibraryProxy::HandleRecvTensorRaw(const RecvTensorRequest *req,
                                           std::function<void(std::vector<zmq::message_t> *, Status)> cb)
{
    d->schedule([this, req, cb]() {
        CallOptions *call_opts = new CallOptions;
        auto resp = new zmq::MultiPartMessage;
        d->m_worker->RecvTensorAsync(call_opts, req, resp, [call_opts, resp, cb](const Status &s) {
            delete call_opts;
            cb(resp->release(), s);
        });
    });
}

} // namespace remote
} // namespace tensorflow
