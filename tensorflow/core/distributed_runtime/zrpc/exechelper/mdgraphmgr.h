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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_MDGRAPHMGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_MDGRAPHMGR_H_

#include "tensorflow/core/distributed_runtime/graph_mgr.h"

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include <functional>

namespace tensorflow {
class Device;
class DeviceMgr;
class FunctionLibraryRuntime;
class NodeDef;
class OpKernel;

struct MultiDeviceExecutorParams;

/**
 * @todo write docs
 */
class MDGraphMgr : public GraphMgr
{
public:
    using ExecutorFactory = std::function<Status(const MultiDeviceExecutorParams &params, const Graph *graph,
                                                 Executor **executor)>;

    explicit MDGraphMgr(const WorkerEnv *env, DeviceMgr *device_mgr);
    ~MDGraphMgr() override;

    // This is not thread safe, must be called before first call of InitItem
    void setExecutorFactory(ExecutorFactory f)
    {
        m_execFactory = f;
    }

protected:
    Status InitItem(const string &session, const GraphDef &gdef, const GraphOptions &graph_options,
                    Item *item) override;

private:
    ExecutorFactory m_execFactory;

    // Global Resource manager shared by all local devices.
    // NOTE: Must be valid when destorying opsegment, because
    // some op uses this during deconstruction.
    std::unique_ptr<ResourceMgr> m_resourceMgr;

    // Global Opsegment shared by all local devices on all workers
    // (we have one and only one local worker)
    std::unique_ptr<OpSegment> m_opseg;

    // Kernel to device name map
    std::unordered_map<const OpKernel*, std::string> m_kernelToDevice;
    mutex m_mu;
};

struct MultiDeviceExecutorParams
{
    std::string session;

    // The devices this executor should use.
    DeviceMgr *deviceMgr;

    // The resource manager this executor should use.
    ResourceMgr *resourceMgr;

    // create_fruntime creates function library runtime given device,
    // caller takes the ownership of the created library runtime.
    std::function<FunctionLibraryRuntime *(Device *)> create_fruntime;
    std::function<void(FunctionLibraryRuntime *)> delete_fruntime;

    // find_kernel returns an instance of op kernel, which was created on device.
    // create_kernel returns an instance of op kernel based on NodeDef for device d.
    // delete_kernel is called for every kernel used by the executor
    // when the executor is deleted.
    std::function<Status(const NodeDef &, std::string *, OpKernel **)> find_kernel;

    std::function<Status(const NodeDef &, FunctionLibraryRuntime *, OpKernel **)> create_kernel;

    std::function<void(OpKernel *, FunctionLibraryRuntime *)> delete_kernel;

    Executor::Args::NodeOutputsCallback node_outputs_cb;
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_MDGRAPHMGR_H_
