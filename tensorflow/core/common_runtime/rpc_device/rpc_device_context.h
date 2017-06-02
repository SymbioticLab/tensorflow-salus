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

#ifndef RPCDEVICECONTEXT_H
#define RPCDEVICECONTEXT_H

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class RpcClient;

/**
 * @todo write docs
 */
class RpcDeviceContext : public DeviceContext
{
public:
    RpcDeviceContext(RpcClient &client);

    ~RpcDeviceContext() override;

    void CopyCPUTensorToDevice(const Tensor *cpu_tensor, Device *device,
                               Tensor *device_tensor,
                               StatusCallback done) const override;

    void CopyDeviceTensorToCPU(const Tensor *device_tensor, StringPiece edge_name,
                               Device *device, Tensor *cpu_tensor,
                               StatusCallback done) override;
private:
    RpcClient &m_rpc;
};

}
#endif // RPCDEVICECONTEXT_H
