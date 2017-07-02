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

#include "exechelpers.h"

namespace tensorflow {

std::unique_ptr<Graph> ExecHelpers::convertGraphDefToGraph(const GraphDef &graphdef,
                                                           const FunctionLibraryDefinition *fdef,
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
        for (auto node: graph->nodes()) {
            LOG(INFO) << "Procssing node id" << node->id();
            gindex[node->name()] = node->id();
        }
        LOG(INFO) << "gindex built " << ok;
    }
    return graph;
}

}
