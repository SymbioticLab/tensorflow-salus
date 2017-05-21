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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

RpcDeviceContext::RpcDeviceContext()
{

}

RpcDeviceContext::~RpcDeviceContext()
{

}

void RpcDeviceContext::CopyCPUTensorToDevice(const Tensor *cpu_tensor, Device *device, Tensor *device_tensor,
                                             StatusCallback done) const {
    LOG(INFO) << "RpcDeviceContext::CopyCPUTensorToDevice";
    const int64 total_bytes = cpu_tensor->TotalBytes();
    if (total_bytes > 0) {
        const void *src_ptr = DMAHelper::base(cpu_tensor);
        void *dst_ptr = DMAHelper::base(device_tensor);
        // TODO: copy data from CPU to RPC device
        switch (cpu_tensor->dtype()) {
        case DT_FLOAT:
            break;
        case DT_DOUBLE:
            break;
        case DT_INT32:
            break;
        case DT_INT64:
            break;
        case DT_HALF:
            break;
        case DT_COMPLEX64:
            break;
        case DT_COMPLEX128:
            break;
        case DT_INT8:
            break;
        case DT_INT16:
            break;
        case DT_UINT8:
            break;
        case DT_UINT16:
            break;
        case DT_BOOL:
            break;
        default:
            assert(false && "unsupported type");
        }
    }
    done(Status::OK());
}

void RpcDeviceContext::CopyDeviceTensorToCPU(const Tensor *device_tensor, StringPiece edge_name,
                                             Device *device, Tensor *cpu_tensor, StatusCallback done) {
    LOG(INFO) << "RpcDeviceContext::CopyDeviceTensorToCPU";
    const int64 total_bytes = device_tensor->TotalBytes();
    if (total_bytes > 0) {
        const void *src_ptr = DMAHelper::base(device_tensor);
        void *dst_ptr = DMAHelper::base(cpu_tensor);
        switch (device_tensor->dtype()) {
            // TODO: copy data from RPC device to CPU
        case DT_FLOAT:
            break;
        case DT_DOUBLE:
            break;
        case DT_INT32:
            break;
        case DT_INT64:
            break;
        case DT_HALF:
            break;
        case DT_COMPLEX64:
            break;
        case DT_COMPLEX128:
            break;
        case DT_INT8:
            break;
        case DT_INT16:
            break;
        case DT_UINT8:
            break;
        case DT_UINT16:
            break;
        case DT_BOOL:
            break;
        default:
            assert(false && "unsupported type");
        }
    }
    done(Status::OK());
}

} // namespace tensorflow
