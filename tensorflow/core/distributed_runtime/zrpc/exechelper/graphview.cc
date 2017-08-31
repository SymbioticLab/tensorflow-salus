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

#include "graphview.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace remote {

std::unique_ptr<Graph> ExecHelpers::convertGraphDefToGraph(
    const GraphDef &graphdef, const FunctionLibraryDefinition *fdef,
    std::unordered_map<std::string, int> &gindex)
{
    auto graph = std::unique_ptr<Graph>(new Graph(fdef));
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    auto ok = ConvertGraphDefToGraph(opts, graphdef, graph.get());
    LOG(INFO) << "ConvertGraphDefToGraph returned " << ok;
    if (!ok.ok()) {
        graph.reset();
    } else {
        for (auto node : graph->nodes()) {
            LOG(INFO) << "Procssing node id" << node->id();
            gindex[node->name()] = node->id();
        }
        LOG(INFO) << "gindex built " << ok;
    }
    return graph;
}

GraphView::~GraphView()
{
    static_assert(std::is_trivially_destructible<AllocatorAttributes>::value,
                  "Update code if AllocatorAttributes gains a destructor");
    static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                  "Update code if EdgeInfo gains a destructor");
    for (int i = 0; i < num_nodes_; i++) {
        NodeItem *n = node(i);
        if (n != nullptr) {
            n->NodeItem::~NodeItem();
            // Memory for "n" itself is held in space_ & gets cleaned up below
        }
    }
    delete[] node_offsets_;
    delete[] space_;
}

size_t GraphView::NodeItemBytes(const Node *n)
{
    const int num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    // Compute number of bytes needed for NodeItem and variable length data.
    // We do not subtract sizeof(var) since num_inputs/num_outputs might
    // both be zero.
    const size_t raw_bytes = sizeof(NodeItem)                            // Fixed
                             + num_output_edges * sizeof(EdgeInfo)       // output_edges[...]
                             + num_outputs * sizeof(AllocatorAttributes) // output_attr[...]
                             + num_inputs * sizeof(uint8)                // input_type[num_inputs]
                             + num_outputs * sizeof(uint8);              // output_type[num_outputs]
    static constexpr size_t kItemAlignment = sizeof(NodeItem *);
    static_assert(kItemAlignment % alignof(NodeItem) == 0,
                  "NodeItem must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(EdgeInfo) == 0,
                  "EdgeInfo must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(AllocatorAttributes) == 0,
                  "AllocatorAttributes must be aligned with kItemAlignment");
    static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0,
                  "NodeItem must be aligned with EdgeInfo");
    static_assert(sizeof(NodeItem) % alignof(AllocatorAttributes) == 0,
                  "NodeItem must be aligned with AllocatorAttributes");
    static_assert(sizeof(EdgeInfo) % alignof(AllocatorAttributes) == 0,
                  "EdgeInfo must be aligned with AllocatorAttributes");
    const size_t bytes = ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
    return bytes;
}

char *GraphView::InitializeNode(char *ptr, const Node *n)
{
    const int id = n->id();
    CHECK(node_offsets_[id] == kuint32max); // Initial value in constructor

    const size_t bytes = NodeItemBytes(n);
    constexpr size_t kItemAlignment = sizeof(NodeItem *);
    CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
    NodeItem *item = reinterpret_cast<NodeItem *>(ptr);

    // We store a 32-bit offset relative to the beginning of space_, so that we
    // only need an array of 32-bit values to map from node id to the NodeItem*,
    // (versus 64 bits on most machines if we just stored an array of NodeItem*
    // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
    // values as "int" vs "size_t" in CHECK_LE.
    CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
    const uint32 offset = ptr - space_;
    node_offsets_[id] = offset;
    ptr += bytes;

    const int num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    new (item) NodeItem();
    item->num_inputs = num_inputs;
    item->num_outputs = num_outputs;
    item->num_output_edges = num_output_edges;

    // Fill output edges.
    // Keep track of the last EdgeInfo in the EdngeInfo array that references
    // a given output slot.  For all but the last, we need to do a copy of the
    // Tensor when propagating results downstream in the graph, but for the
    // last one, we can just do a move of the Tensor object to propagate it.
    gtl::InlinedVector<EdgeInfo *, 4> last_indices(num_outputs, nullptr);
    EdgeInfo *dst_edge = item->output_edge_base();
    for (auto e : n->out_edges()) {
        dst_edge->dst_id = e->dst()->id();
        CHECK_LE(e->src_output(), ((int32) 0x3FFFFFFF)); // Must fit in 31 bits
        dst_edge->output_slot = e->src_output();
        dst_edge->is_last = false;
        const int output_slot = dst_edge->output_slot;
        if (output_slot >= 0) {
            last_indices[output_slot] = dst_edge;
        }
        dst_edge->input_slot = e->dst_input();
        dst_edge++;
    }
    for (EdgeInfo *edge_info : last_indices) {
        if (edge_info != nullptr) {
            edge_info->is_last = true;
        }
    }

    AllocatorAttributes *output_attrs = item->output_attr_base();
    for (int i = 0; i < num_outputs; i++) {
        new (&output_attrs[i]) AllocatorAttributes();
    }

    DCHECK_LT(DataType_MAX, 255); // Must fit in uint8
    uint8 *input_types = item->input_type_base();
    for (int i = 0; i < num_inputs; i++) {
        input_types[i] = static_cast<uint8>(n->input_type(i));
        DCHECK_EQ(item->input_type(i), n->input_type(i));
    }

    uint8 *output_types = item->output_type_base();
    for (int i = 0; i < num_outputs; i++) {
        output_types[i] = static_cast<uint8>(n->output_type(i));
        DCHECK_EQ(item->output_type(i), n->output_type(i));
    }
    return ptr;
}

void GraphView::Initialize(const Graph *g)
{
    CHECK(node_offsets_ == nullptr);
    const int num_nodes = g->num_node_ids();
    num_nodes_ = num_nodes;
    size_t total_bytes = 0;
    for (const Node *n : g->nodes()) {
        total_bytes += NodeItemBytes(n);
    }

    node_offsets_ = new uint32[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        node_offsets_[i] = kuint32max;
    }

    space_ = new char[total_bytes]; // NodeItem objects are allocated here
    char *ptr = space_;
    for (const Node *n : g->nodes()) {
        ptr = InitializeNode(ptr, n);
    }
    CHECK_EQ(ptr, space_ + total_bytes);
}

Status InferAllocAttr(const Node *n, const Node *dst,
                      const DeviceNameUtils::ParsedName &local_dev_name, AllocatorAttributes *attr)
{
    Status s;
    // Note that it's possible for *n to be a Recv and *dst to be a Send,
    // so these two cases are not mutually exclusive.
    if (IsRecv(n)) {
        string src_name;
        s = GetNodeAttr(n->def(), "send_device", &src_name);
        if (!s.ok())
            return s;
        DeviceNameUtils::ParsedName parsed_src_name;
        if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
            s = errors::Internal("Bad send_device attr '", src_name, "' in node ", n->name());
            return s;
        }
        if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
            // Value is going to be the sink of an RPC.
            attr->set_nic_compatible(true);
            VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
        } else if ((local_dev_name.type == "CPU" || n->IsHostRecv())
                   && parsed_src_name.type != "CPU") {
            // Value is going to be the sink of a local DMA from GPU to CPU (or other
            // types of accelerators).
            attr->set_gpu_compatible(true);
            VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
        } else {
            VLOG(2) << "default alloc case local type " << local_dev_name.type << " remote type "
                    << parsed_src_name.type;
        }
    }
    if (IsSend(dst)) {
        string dst_name;
        s = GetNodeAttr(dst->def(), "recv_device", &dst_name);
        if (!s.ok())
            return s;
        DeviceNameUtils::ParsedName parsed_dst_name;
        if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
            s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ", n->name());
            return s;
        }
        if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
            // Value is going to be the source of an RPC.
            attr->set_nic_compatible(true);
            VLOG(2) << "node " << n->name() << " is the source of an RPC out";
        } else if ((local_dev_name.type == "CPU" || dst->IsHostSend())
                   && parsed_dst_name.type != "CPU") {
            // Value is going to be the source of a local DMA from CPU to GPU (or
            // other types of accelerators).
            // Note that this does not cover the case where the allocation of the
            // output tensor is not generated by the src: n.
            attr->set_gpu_compatible(true);
            VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
        } else {
            VLOG(2) << "default alloc case local type " << local_dev_name.type << " remote type "
                    << parsed_dst_name.type;
        }
    }
    return s;
}

Status GraphView::SetAllocAttrForNode(const Node *n, const Device *device, const OpKernel *op_kernel) const
{
    Status s;
    DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

    NodeItem *item = node(n->id());
    AllocatorAttributes *attrs = item->output_attr_base();

    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (auto e : n->out_edges()) {
        if (!e->IsControlEdge()) {
            AllocatorAttributes attr;
            s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
            if (!s.ok())
                return s;
            if (attr.value != 0) {
                attrs[e->src_output()].Merge(attr);
            }
        }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
        DCHECK_LT(out, op_kernel->output_memory_types().size());
        bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
        if (on_host) {
            AllocatorAttributes h;
            h.set_on_host(on_host);
            attrs[out].Merge(h);
        }
    }
    return s;
}

} // namespace remote
} // namespace tensorflow
