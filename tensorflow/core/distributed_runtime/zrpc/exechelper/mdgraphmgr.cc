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

#include "mdgraphmgr.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/graph/validate.h"

namespace tensorflow {

MDGraphMgr::MDGraphMgr(const WorkerEnv *env, DeviceMgr *device_mgr)
    : GraphMgr(env, device_mgr)
    , m_execFactory(nullptr)
    , m_resourceMgr(new ResourceMgr)
    , m_opseg(new OpSegment)
{
}

MDGraphMgr::~MDGraphMgr()
{
    m_opseg.reset();

    m_resourceMgr.reset();
}

Status MDGraphMgr::InitItem(const string &session, const GraphDef &gdef, const GraphOptions &graph_options,
                            Item *item)
{
    DCHECK(m_execFactory);

    item->session = session;
    item->lib_def.reset(new FunctionLibraryDefinition(OpRegistry::Global(), gdef.library()));

    //   TF_RETURN_IF_ERROR(ValidateGraphDefForDevices(gdef));

    if (gdef.versions().producer() >= 5) {
        // Validate the graph: we assume that merging two valid graphs
        // should maintain graph validity.
        TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *item->lib_def));
    }

    // Constructs the graph out of "gdef".
    Graph graph(item->lib_def);
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = true;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, gdef, &graph));

    // Splits "graph" into multiple subgraphs by device names.
    std::unordered_map<string, GraphDef> partitions;
    PartitionOptions popts;
    popts.node_to_loc = [](const Node *node) { return node->assigned_device_name(); };
    popts.new_name = [this](const string &prefix) {
        mutex_lock l(mu_);
        return strings::StrCat(prefix, "_G", next_id_++);
    };
    popts.get_incarnation = [this](const string &name) -> int64 {
        Device *device = nullptr;
        Status s = worker_env_->device_mgr->LookupDevice(name, &device);
        if (s.ok()) {
            return device->attributes().incarnation();
        } else {
            return PartitionOptions::kIllegalIncarnation;
        }
    };
    popts.control_flow_added = true;
    popts.scheduling_for_recvs = graph_options.enable_recv_scheduling();
    TF_RETURN_IF_ERROR(Partition(popts, &graph, &partitions));
    if (popts.scheduling_for_recvs) {
        TF_RETURN_IF_ERROR(AddControlEdges(popts, &partitions));
    }

    std::unordered_map<string, std::unique_ptr<Graph>> partition_graphs;
    for (const auto &partition : partitions) {
        std::unique_ptr<Graph> device_graph(new Graph(item->lib_def));
        GraphConstructorOptions device_opts;
        // There are internal operations (e.g., send/recv) that we now allow.
        device_opts.allow_internal_ops = true;
        device_opts.expect_device_spec = true;
        TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(device_opts, partition.second, device_graph.get()));
        partition_graphs.emplace(partition.first, std::move(device_graph));
    }

    GraphOptimizationPassOptions optimization_options;
    optimization_options.flib_def = item->lib_def;
    optimization_options.partition_graphs = &partition_graphs;
    TF_RETURN_IF_ERROR(
        OptimizationPassRegistry::Global()->RunGrouping(OptimizationPassRegistry::POST_PARTITIONING,
                                                        optimization_options));

    MultiDeviceExecutorParams params;
    params.session = session;
    params.deviceMgr = worker_env_->device_mgr;
    params.resourceMgr = m_resourceMgr.get();

    item->units.reserve(partitions.size());
    item->graph_mgr = this;
    const auto &optimizer_opts = graph_options.optimizer_options();
    GraphOptimizer optimizer(optimizer_opts);
    for (auto &p : partition_graphs) {
        const string &device_name = p.first;
        std::unique_ptr<Graph> &subgraph = p.second;
        item->units.resize(item->units.size() + 1);
        ExecutionUnit *unit = &(item->units.back());

        // Find the device.
        Status s = worker_env_->device_mgr->LookupDevice(device_name, &unit->device);
        if (!s.ok()) {
            // Remove the empty unit from the item as the item destructor wants all
            // units to have valid devices.
            item->units.pop_back();
            return s;
        }

        // Give the device an opportunity to rewrite its subgraph.
        TF_RETURN_IF_ERROR(unit->device->MaybeRewriteGraph(gdef.library(), &subgraph));

        // Top-level nodes in the graph uses the op segment to cache
        // kernels. Therefore, as long as the executor is alive, we need
        // to ensure the kernels cached for the session are alive.
        //     auto opseg = unit->device->op_segment();
        auto opseg = m_opseg.get();
        opseg->AddHold(session);

        auto producer = subgraph->versions().producer();
        auto worker_env = worker_env_;
        params.create_fruntime = [worker_env, producer, item, optimizer_opts](Device *dev) {
            item->Ref();
            return NewFunctionLibraryRuntime(worker_env->device_mgr, worker_env->env, dev, producer,
                                             item->lib_def, optimizer_opts);
        };

        params.delete_fruntime = [item] (FunctionLibraryRuntime *r) {
            delete r;
            item->Unref();
        };

        // Construct the root executor for the subgraph.
        params.find_kernel = [this, session, opseg](const NodeDef &ndef, std::string *devName,
                                                    OpKernel **kernel) {
            *kernel = nullptr;
            devName->clear();

            bool found = true;
            auto ok = opseg->FindOrCreate(session, ndef.name(), kernel, [&found](OpKernel **) {
                found = false;
                return Status::OK();
            });
            if (!ok.ok() || !found) {
                return ok;
            }

            mutex_lock l(m_mu);
            auto it = m_kernelToDevice.find(*kernel);
            if (it == m_kernelToDevice.end()) {
                return errors::Internal("We've created the kernel, but don't remember its device");
            }
            *devName = it->second;
            return Status::OK();
        };

        params.create_kernel = [this, session, opseg](const NodeDef &ndef,
                                                      FunctionLibraryRuntime *lib,
                                                      OpKernel **kernel) -> Status {
            // Caches the kernel only if the node is stateful.
            if (!lib->IsStateful(ndef.op())) {
                return lib->CreateKernel(ndef, kernel);
            }
            auto create_fn = [this,lib, &ndef](OpKernel **kernel) {
                auto s = lib->CreateKernel(ndef, kernel);
                mutex_lock l(m_mu);
                m_kernelToDevice[*kernel] = lib->device()->name();
                return s;
            };
            // Kernels created for subgraph nodes need to be cached.  On
            // cache miss, create_fn() is invoked to create a kernel based
            // on the function library here + global op registry.
            return opseg->FindOrCreate(session, ndef.name(), kernel, create_fn);
        };

        params.delete_kernel = [](OpKernel *kernel, FunctionLibraryRuntime *lib) {
            // If the node is stateful, opseg owns it. Otherwise, delete it.
            if (kernel && !lib->IsStateful(kernel->type_string())) {
                delete kernel;
            }
        };

        unit->lib = NewFunctionLibraryRuntime(worker_env_->device_mgr, worker_env_->env, unit->device,
                                              subgraph->versions().producer(), item->lib_def,
                                              graph_options.optimizer_options());

        optimizer.Optimize(unit->lib, worker_env_->env, unit->device, &subgraph);
        TF_RETURN_IF_ERROR(
            EnsureMemoryTypes(DeviceType(unit->device->device_type()), unit->device->name(), subgraph.get()));
        unit->graph = subgraph.get();
        unit->build_cost_model = graph_options.build_cost_model();

        // TODO: Always skip cost models, which causes a deadlock
        // when calling item->Unref() from delete_fruntime.
        skip_cost_models_ = true;

        TF_RETURN_IF_ERROR(m_execFactory(params, subgraph.release(), &unit->root));
    }
    return Status::OK();
}

} // namespace tensorflow
