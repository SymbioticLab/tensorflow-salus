/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/distributed_runtime/zrpc/zrpc_remote_master.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_master_service_stub.h"
#include "tensorflow/core/distributed_runtime/zrpc/zrpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// ZrpcRemoteMaster is an implementation of the MasterInterface
// that uses ZeroMQ to talk to the Master service.
class ZrpcRemoteMaster : public MasterInterface
{
public:
    explicit ZrpcRemoteMaster(Env *env, const std::string &endpoint)
        : stub_(env, endpoint)
    {
    }

    ~ZrpcRemoteMaster() override
    {
    }

    Status CreateSession(CallOptions *call_options, const CreateSessionRequest *request,
                         CreateSessionResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        auto s = stub_.CreateSession(*request, response);
        LOG(INFO) << "RpcClient created session with id " << response->session_handle();
        return s;
    }

    Status ExtendSession(CallOptions *call_options, const ExtendSessionRequest *request,
                         ExtendSessionResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.ExtendSession(*request, response);
    }

    Status PartialRunSetup(CallOptions *call_options, const PartialRunSetupRequest *request,
                           PartialRunSetupResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.PartialRunSetup(*request, response);
    }

    Status RunStep(CallOptions *call_options, RunStepRequestWrapper *request,
                   MutableRunStepResponseWrapper *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.RunStep(request->ToProto(), get_proto_from_wrapper(response));
    }

    Status CloseSession(CallOptions *call_options, const CloseSessionRequest *request,
                        CloseSessionResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.CloseSession(*request, response);
    }

    Status ListDevices(CallOptions *call_options, const ListDevicesRequest *request,
                       ListDevicesResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.ListDevices(*request, response);
    }

    Status Reset(CallOptions *call_options, const ResetRequest *request, ResetResponse *response) override
    {
        SetDeadline(call_options->GetTimeout());
        return stub_.Reset(*request, response);
    }

private:
    ZrpcMasterServiceStub stub_;

    void SetDeadline(int64 time_in_ms)
    {
        if (time_in_ms > 0) {
            LOG(WARNING) << "Setting deadline is not supported in Zrpc";
            // ctx->set_deadline(gpr_time_from_millis(time_in_ms, GPR_TIMESPAN));
        }
    }
};

MasterInterface *NewZrpcRemoteMaster(Env *env, const std::string &endpoint)
{
    return new ZrpcRemoteMaster(env, endpoint);
}

} // namespace tensorflow
