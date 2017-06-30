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

#ifndef EXECHELPERS_H
#define EXECHELPERS_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include <memory>

namespace tensorflow {
class Graph;

/**
 * @todo write docs
 */
class ExecHelpers
{
public:
    static std::unique_ptr<Graph> convertGraphDefToGraph(const GraphDef &graphdef,
                                                         const FunctionLibraryDefinition *fdef);
};

} // namespace tensorflow

#endif // EXECHELPERS_H
