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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/rpc_device/rpc/rpcclient.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

RPCDeviceContext::RPCDeviceContext(RpcClient &client)
    : m_rpc(client)
{ }

RPCDeviceContext::~RPCDeviceContext()
{

}

void RPCDeviceContext::CopyCPUTensorToDevice(const Tensor *cpu_tensor, Device *device, Tensor *device_tensor,
                                             StatusCallback done) const {
    LOG(INFO) << "RpcDeviceContext::CopyCPUTensorToDevice";
    auto status = m_rpc.push(device_tensor, cpu_tensor);
    done(status);
}

void RPCDeviceContext::CopyDeviceTensorToCPU(const Tensor *device_tensor, StringPiece edge_name,
                                             Device *device, Tensor *cpu_tensor, StatusCallback done) {
    LOG(INFO) << "RpcDeviceContext::CopyDeviceTensorToCPU";

    auto status = m_rpc.fetch(cpu_tensor, device_tensor);

    done(status);
}

} // namespace tensorflow
