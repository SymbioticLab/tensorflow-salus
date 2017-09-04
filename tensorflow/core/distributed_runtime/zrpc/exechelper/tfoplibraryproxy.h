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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_TFOPLIBRARYPROXY_H
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_TFOPLIBRARYPROXY_H

#include "tensorflow/core/lib/core/status.h"

#include <memory>
#include <functional>
#include <vector>

#define CallWithMasterMethodName(m) \
    m(CreateSession) \
    m(ExtendSession) \
    m(PartialRunSetup) \
    m(CloseSession) \
    m(ListDevices) \
    m(Reset)

#define CallWithWorkerMethodName(m) \
    m(GetStatus) \
    m(RegisterGraph) \
    m(DeregisterGraph) \
    m(CleanupGraph) \
    m(CleanupAll) \
    m(Logging) \
    m(Tracing)

#define CallWithAllMethodName(m) \
    CallWithMasterMethodName(m) \
    m(RunStep) \
    CallWithWorkerMethodName(m) \
    m(RunGraph) \
    m(RecvTensor)

namespace zmq {
class message_t;
}

namespace tensorflow {

#define FWD_DECLARE(name) \
    class name ## Request; \
    class name ## Response;

CallWithAllMethodName(FWD_DECLARE)

#undef FWD_DECLARE

struct MultiDeviceExecutorParams;
class Graph;
class Executor;
class Env;
class Device;
class DeviceMgr;
class ConfigProto;

namespace remote {

class TFSessionProxy;
class TFOpLibraryProxy
{
public:
    using ExecutorFactory =
        std::function<Status(const MultiDeviceExecutorParams &params, const Graph *graph, Executor **executor)>;
    /**
     * Default constructor
     */
    explicit TFOpLibraryProxy(ExecutorFactory execFactory);
    ~TFOpLibraryProxy();

    Status globalInit(const ConfigProto &options);
    Status newSession(std::unique_ptr<TFSessionProxy> &p);

private:
    friend class TFSessionProxy;

    Env *m_env;
    ExecutorFactory m_execFactory;
    std::unique_ptr<DeviceMgr> m_deviceMgr;
    std::vector<Device *> m_devices;
};

/**
 * Hides all implementations in tensorflow dynamic library in a d-pointer,
 * so this is the only class exposed to the outside
 */
class TFSessionProxyPrivate;
class TFSessionProxy
{
public:
    TFSessionProxy();
    ~TFSessionProxy();

    /**
     * Move constructor
     */
    TFSessionProxy(TFSessionProxy &&other);

    Status init(TFOpLibraryProxy *proxy);

    void schedule(std::function<void()> f);

#define DECLARE_HANDLER(name) \
    void Handle ## name (const name ## Request *req, std::function<void(name ## Response*, Status)> cb);

    CallWithMasterMethodName(DECLARE_HANDLER)
    DECLARE_HANDLER(RunStep)
    CallWithWorkerMethodName(DECLARE_HANDLER)
    DECLARE_HANDLER(RunGraph)

#undef DECLARE_HANDLER

    void HandleRecvTensorRaw(const RecvTensorRequest *req, std::function<void(std::vector<zmq::message_t>*, Status)> cb);

private:
    std::unique_ptr<TFSessionProxyPrivate> d;
};

} // namespace remote
} // namespace tensorflow

#endif // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ZRPC_EXECHELPER_TFOPLIBRARYPROXY_H
