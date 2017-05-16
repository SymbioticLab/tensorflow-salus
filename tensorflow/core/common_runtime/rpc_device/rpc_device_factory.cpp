#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_device.h"
#include "tensorflow/core/common_runtime/rpc_device/threadpiscine_device.h"

namespace tensorflow {

class RpcDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions &options, const string &name_prefix,
                       std::vector<Device *> *devices) override {
    int n = 1;
    auto iter = options.config.device_count().find("RPC");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/device:RPC:", i);
      devices->push_back(
//           new RpcDevice(options, name, Bytes(256 << 20), DeviceLocality(),
//                         RpcLDevice::GetShortDeviceDescription(),
//                         cl::sycl::gpu_selector(), cpu_allocator()));
          new ThreadPiscineDevice(options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
    }
    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("RPC", RpcDeviceFactory, 200);

}  // namespace tensorflow
