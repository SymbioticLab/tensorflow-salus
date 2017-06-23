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
 *
 */

#include "rpc_device_context.h"

#include "tensorflow/core/common_runtime/rpc_device/rpc_allocator.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/rpcclient.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/executor.pb.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/tfoplibrary.pb.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

RPCDeviceContext::RPCDeviceContext(RpcClient &client, const Graph *graph,
                                   const FunctionDefLibrary &fdef, const ConfigProto &cfgProto)
    : m_rpc(client)
{
    DumpGraph("RPCDeviceContext::initialize", graph);
    // Create and cache the GraphDef, and build an index from node name to NodeDef
    // Because node->def() can be out of sync for inputs
    graph->ToGraphDef(&m_graphdef);
    for (int i = 0; i != m_graphdef.node_size(); ++i) {
        auto &ndef = m_graphdef.node(i);
        m_name2defidx[ndef.name()] = i;
        LOG(INFO) << "K->N Mapping: " << ndef.name() << " -> " << i;
    }
    m_rpc.createSession(cfgProto, fdef, m_graphdef, m_sessionId);
}

RPCDeviceContext::~RPCDeviceContext() { }

const NodeDef &RPCDeviceContext::findNodeDefFor(const OpKernel *kernel) const
{
    // NOTE: kernel->def() might be outdated due to optimizations. Find nodedef from m_graphdef
    auto it = m_name2defidx.find(kernel->name());
    if (it != m_name2defidx.end()) {
        return m_graphdef.node(it->second);
    }
    // fallback to node->def(), this may happen for _SOURCE or other special nodes
    return kernel->def();
}

std::string RPCDeviceContext::sessionId() const
{
    return m_sessionId;
}

Tensor RPCDeviceContext::tensorFromProtoMeta(const TensorProto &outdef)
{
    uint64_t addr_handle = 0;
    if (outdef.int64_val_size() == 1) {
        addr_handle = outdef.int64_val(0);
    } else if (outdef.int64_val_size() > 1){
        LOG(ERROR) << "The tensorproto is not a valid protometa: " << outdef.DebugString();
    }

    auto dtype = outdef.dtype();
    if (IsRefType(dtype)) {
        dtype = RemoveRefType(dtype);
    }
    // create a one time allocator, which will delete itself after DeallocateRaw
    auto alloc = OneTimeAllocator::create(addr_handle).release();
    return Tensor(alloc, dtype, TensorShape(outdef.tensor_shape()));
}

void RPCDeviceContext::tensorToProtoMeta(TensorProto *meta, const Tensor &tensor, bool is_ref)
{
    LOG(WARNING) << "Do not use this !!!!!!!!!!!!!!!!!!!!!!";
    auto dtype = tensor.dtype();
    if (is_ref) {
        dtype = MakeRefType(dtype);
    }
    meta->set_dtype(dtype);

    tensor.shape().AsProto(meta->mutable_tensor_shape());

    LOG(INFO) << "Serialize tensor to proto meta, initialized: " << tensor.IsInitialized();
    LOG(INFO) << "Serialize tensor to proto meta, shape.num_elements: " << tensor.shape().num_elements();
    if (tensor.IsInitialized() && tensor.shape().num_elements() > 0) {
        auto addr_handle = reinterpret_cast<uint64_t>(tensor.tensor_data().data());
        // HACK: use a int64 val entry to store the addr handle for simplicity,
        // idealy should store this in tensor_content with proper encoding.
        meta->add_int64_val(addr_handle);
    }
}

void RPCDeviceContext::Compute(OpKernel *op_kernel, OpKernelContext *context)
{
    auto status = m_rpc.run(this, op_kernel, context);

    if (!status.ok()) {
        LOG(ERROR) << "RPC call failed with " << status;
    }

//     op_kernel->Compute(context);

    LOG(INFO) << "context.status() " << context->status();
    LOG(INFO) << "context.is_output_dead() " << *context->is_output_dead();
    LOG(INFO) << "context.num_outputs() " << context->num_outputs();
}

void RPCDeviceContext::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                                    AsyncOpKernel::DoneCallback done)
{
    m_rpc.runAsync(this, op_kernel, context, std::move(done));
}

void RPCDeviceContext::serializeOpKernel(executor::OpKernelDef *def, tensorflow::OpKernel *kernel)
{
    LOG(INFO) << "About to serialize OpKernel: " << kernel->name();

    def->set_id(kernel->name());
    def->set_oplibrary(executor::TENSORFLOW);

    executor::TFOpKernelDef tfdef;

    *tfdef.mutable_nodedef() = findNodeDefFor(kernel);
    LOG(INFO) << "Serialized OpKernel: " << tfdef.nodedef().DebugString();

    tfdef.set_isasync(kernel->AsAsync() != nullptr);

    tfdef.SerializeToString(def->mutable_extra());

    LOG(INFO) << "Done";
}

void RPCDeviceContext::serializeOpContext(executor::OpContextDef *def, OpKernelContext *context)
{
    LOG(INFO) << "About to serialize OpContext";

    executor::TFOpContextDef tfdef;
    auto params = context->params_;

    tfdef.set_step_id(params->step_id);
    tfdef.set_frame_id(params->frame_iter.frame_id);
    tfdef.set_iter_id(params->frame_iter.iter_id);
    tfdef.set_is_input_dead(params->is_input_dead);

    for (int i = 0; i != context->num_inputs(); i++) {
        auto initem = tfdef.add_inputs();
        initem->set_is_ref(context->input_is_ref(i));
        initem->set_name(findNodeDefFor(&context->op_kernel()).input(i));
    }

    tfdef.SerializeToString(def->mutable_extra());

    LOG(INFO) << "Done";
}

class TensorResource : public ResourceBase
{
public:
    explicit TensorResource(const Tensor&t) : m_tensor(t) {}

    string DebugString() override {
        return strings::StrCat(DataTypeString(m_tensor.dtype()), "/", m_tensor.shape().DebugString());
    }

    inline Tensor *tensor() { return &m_tensor; }
    inline mutex *mu() { return &m_mu; }

private:
    ~TensorResource() override {}
    TF_DISALLOW_COPY_AND_ASSIGN(TensorResource);

    Tensor m_tensor;
    mutex m_mu;
};

void RPCDeviceContext::deserializeOpContext(OpKernelContext *context, const executor::OpContextDef *def)
{
    LOG(INFO) << "About to update context";

    executor::TFOpContextUpdate tfdef;
    tfdef.ParseFromString(def->extra());

    // Emit send on rendezvous
    LOG(INFO) << tfdef.rendeztensors_size() << " emits to send for rendezvous";
    for (int i = 0; i != tfdef.rendeztensors_size(); ++i) {
        const auto &outdef = tfdef.rendeztensors(i);
        Rendezvous::Args args;
        // TODO: is it ok to use op device context?
        args.device_context = context->op_device_context();
        args.alloc_attrs.value = outdef.allocattributes();

        LOG(INFO) << "parsedkey is " << outdef.key();
        Rendezvous::ParsedKey parsed;
        auto status = Rendezvous::ParseKey(outdef.key(), &parsed);
        if (!status.ok()) {
            LOG(ERROR) << "Invalid parsekey " << outdef.key()
                       << " for rendezvous received: " << status;
            continue;
        }
        LOG(INFO) << "Tensor proto is " << outdef.val().DebugString();

        // Directly create tensor on CPU
        args.alloc_attrs.set_on_host(true);
        if (parsed.dst.type != "CPU") {
            LOG(ERROR) << "Rendez from RPC to non CPU device is not supported";
            continue;
        }
        Tensor t;
        if (!t.FromProto(cpu_allocator(), outdef.val())) {
            LOG(ERROR) << "Rendezvous tensors invalid";
        }
        LOG(INFO) << "Rendezvous send tensor " << t.DebugString();
        status = context->rendezvous()->Send(parsed, args, t, outdef.isdead());
        LOG(INFO) << "Rendezvous send finished";
        if (!status.ok()) {
            LOG(ERROR) << "Rendezvous send error: " << status;
            continue;
        }
    }

    LOG(INFO) << "Set outputs";
    static std::atomic<int64> counter(0);
    for (int i = 0; i != tfdef.outputs_size(); ++i) {
        const auto &outdef = tfdef.outputs(i);
        auto tensor = tensorFromProtoMeta(outdef.meta());
        if (outdef.is_ref()) {
            auto name = strings::StrCat(outdef.name(), "_", counter.fetch_add(1));
            TensorResource *tr = nullptr;
            auto ok = context->resource_manager()->LookupOrCreate<TensorResource>("rpcclient", name,
                                                                  &tr, [&tensor](TensorResource **tr){
                *tr = new TensorResource(tensor);
                return Status::OK();
            });
            if (!ok.ok()) {
                LOG(ERROR) << "Creation of reftensor resource failed";
                continue;
            }
            // NOTE: This only works for reference on the same device. And should not be used other than in
            // executor.
            context->set_output_ref(i, tr->mu(), tr->tensor());
        } else {
            context->set_output(i, tensor);
        }
    }
    *context->is_output_dead() = tfdef.is_output_dead();

    if (tfdef.status_code() == error::OK) {
        context->SetStatus(Status::OK());
    } else {
        context->SetStatus(Status(tfdef.status_code(), tfdef.status_msg()));
    }

    LOG(INFO) << "Done";
}

} // namespace tensorflow
