#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_device.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/zmqrpcclient.h"
#include "tensorflow/core/common_runtime/rpc_device/threadpiscine_device.h"

namespace tensorflow {

class RPCDeviceFactory : public DeviceFactory {
public:
    Status CreateDevices(const SessionOptions &options, const string &name_prefix,
                         std::vector<Device *> *devices) override {
        static ZmqRpcClient rpc(options.env, "tcp://localhost:5501");

        int n = 1;
        auto iter = options.config.device_count().find("RPC");
        if (iter != options.config.device_count().end()) {
            n = iter->second;
        }
        for (int i = 0; i < n; i++) {
            string name = strings::StrCat(name_prefix, "/device:RPC:", i);

            auto allocator = new RPCAllocator(rpc);
            auto device = new RPCDevice(options, name, Bytes(256 << 20), DeviceLocality(), allocator, rpc);
            devices->push_back(device);
//             devices->push_back(
//                 new ThreadPiscineDevice(options, name, Bytes(256 << 20), DeviceLocality(), cpu_allocator()));
        }
        return Status::OK();
    }
};

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_RPC, RPCDeviceFactory, 200);

}  // namespace tensorflow
