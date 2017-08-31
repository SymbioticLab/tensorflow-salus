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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_GRAPH_VIEW_H_

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/common_runtime/pending_counts.h"

#include <memory>
#include <unordered_map>

namespace tensorflow {
class Graph;
class OpKernel;

namespace remote {

/**
 * @todo write docs
 */
class ExecHelpers
{
public:
    static std::unique_ptr<Graph> convertGraphDefToGraph(
        const GraphDef &graphdef, const FunctionLibraryDefinition *fdef,
        std::unordered_map<std::string, int> &gindex);
};

class GraphView;

struct EdgeInfo
{
    int dst_id;
    int output_slot : 31;
    // true if this is the last info for output_slot in the EdgeInfo list.
    bool is_last : 1;
    int input_slot;
};

struct NodeItem
{
    NodeItem()
    {
    }

    // A graph node.
    const Node *node = nullptr;

    bool kernel_is_expensive : 1; // True iff kernel->IsExpensive()
    bool is_merge : 1;            // True iff IsMerge(node)
    bool is_enter : 1;            // True iff IsEnter(node)
    bool is_exit : 1;             // True iff IsExit(node)
    bool is_control_trigger : 1;  // True iff IsControlTrigger(node)
    bool is_sink : 1;             // True iff IsSink(node)
    // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
    bool is_enter_exit_or_next_iter : 1;

    // Cached values of node->num_inputs() and node->num_outputs(), to
    // avoid levels of indirection.
    int num_inputs;
    int num_outputs;

    // ExecutorImpl::tensors_[input_start] is the 1st positional input
    // for this node.
    int input_start = 0;

    // Number of output edges.
    int num_output_edges;

    PendingCounts::Handle pending_id;

    const EdgeInfo *output_edge_list() const
    {
        return output_edge_base();
    }

    // ith output edge.
    const EdgeInfo &output_edge(int i) const
    {
        DCHECK_GE(i, 0);
        DCHECK_LT(i, num_output_edges);
        return output_edge_base()[i];
    }

    DataType input_type(int i) const
    {
        DCHECK_LT(i, num_inputs);
        return static_cast<DataType>(input_type_base()[i]);
    }
    DataType output_type(int i) const
    {
        DCHECK_LT(i, num_outputs);
        return static_cast<DataType>(output_type_base()[i]);
    }

    // Return array of per-output allocator attributes.
    const AllocatorAttributes *output_attrs() const
    {
        return output_attr_base();
    }

private:
    friend class GraphView;

    // Variable length section starts immediately after *this
    // (uint8 is enough for DataType).
    //   EdgeInfo            out_edges[num_out_edges];
    //   AllocatorAttributes output_attr[num_outputs];
    //   uint8               input_type[num_inputs];
    //   uint8               output_type[num_outputs];

    // Return pointer to variable length section.
    char *var() const
    {
        return const_cast<char *>(reinterpret_cast<const char *>(this) + sizeof(NodeItem));
    }

    EdgeInfo *output_edge_base() const
    {
        return reinterpret_cast<EdgeInfo *>(var());
    }
    AllocatorAttributes *output_attr_base() const
    {
        return reinterpret_cast<AllocatorAttributes *>(var() + sizeof(EdgeInfo) * num_output_edges);
    }
    uint8 *input_type_base() const
    {
        return reinterpret_cast<uint8 *>(var() + sizeof(EdgeInfo) * num_output_edges
                                         + sizeof(AllocatorAttributes) * num_outputs);
    }
    uint8 *output_type_base() const
    {
        return reinterpret_cast<uint8 *>(var() + sizeof(EdgeInfo) * num_output_edges
                                         + sizeof(AllocatorAttributes) * num_outputs
                                         + sizeof(uint8) * num_inputs);
    }

    TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};

// Immutable view of a Graph organized for efficient execution.
class GraphView
{
public:
    GraphView()
        : space_(nullptr)
    {
    }
    ~GraphView();

    void Initialize(const Graph *g);
    //     Status SetAllocAttrs(const Graph* g, const Device* device);
    Status SetAllocAttrForNode(const Node *n, const Device *device, const OpKernel *op_kernel) const;

    NodeItem *node(int id) const
    {
        DCHECK_GE(id, 0);
        DCHECK_LT(id, num_nodes_);
        uint32 offset = node_offsets_[id];
        return ((offset == kuint32max) ? nullptr :
                                         reinterpret_cast<NodeItem *>(space_ + node_offsets_[id]));
    }

private:
    char *InitializeNode(char *ptr, const Node *n);
    size_t NodeItemBytes(const Node *n);

    int32 num_nodes_ = 0;
    uint32 *node_offsets_ = nullptr; // array of size "graph_.num_node_ids()"
    // node_offsets_[id] holds the byte offset for node w/ "id" in space_

    char *space_; // NodeItem objects are allocated here

    TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};

} // namespace remote
} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_GRAPH_VIEW_H_
