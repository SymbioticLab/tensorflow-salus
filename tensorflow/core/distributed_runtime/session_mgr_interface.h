/*
 * <one line to give the library's name and an idea of what it does.>
 * Copyright (C) 2018  Peifeng Yu <peifeng@umich.edu>
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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_INTERFACE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_INTERFACE_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

#include <functional>
#include <string>

namespace tensorflow {

class WorkerSession;

class SessionMgrInterface {
 public:
  virtual ~SessionMgrInterface() = default;

  // Allocates state for a new session.
  virtual Status CreateSession(const string& session, const ServerDef& server_def, bool isolate_session_state) = 0;

  // Locates the worker session for a given session handle
  virtual WorkerSession* WorkerSessionForSession(const string& session) = 0;

  virtual Status DeleteSession(const string& session) = 0;
};

} // namespace tensorflow

#endif // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SESSION_MGR_INTERFACE_H_
