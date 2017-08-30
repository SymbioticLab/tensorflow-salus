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

#include <functional>

namespace tensorflow {
/**
 * @todo write docs
 */
class MDGraphMgr : public GraphMgr
{
public:
    using ExecutorFactory =
        std::function<Status(const LocalExecutorParams &params, const Graph *graph, Executor **executor)>;

    MDGraphMgr(const WorkerEnv *env, ExecutorFactory execFactory);
    ~MDGraphMgr() override;

protected:
    Status InitItem(const string &session, const GraphDef &gdef, const GraphOptions &graph_options,
                    Item *item) override;

private:
    ExecutorFactory m_execFactory;
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_MDGRAPHMGR_H_
