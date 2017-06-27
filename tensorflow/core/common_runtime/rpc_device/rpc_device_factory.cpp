// Force to use non-debug version of protobuf map, which changes its hashing function
// according to debug state, causing problems when two libraries both use protobuf, but
// only one of them is built with debug. Then passing a map from one library to the other
// becomes impossible because values inserted using one hashing function can't be found
// using another hashing function.
#ifdef NDEBUG
#undef NDEBUG
#include "google/protobuf/map.h"
#define NDEBUG
#else
#include "google/protobuf/map.h"
#endif

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc_device.h"
#include "tensorflow/core/common_runtime/rpc_device/rpc/zmqrpcclient.h"
#include "tensorflow/core/common_runtime/rpc_device/threadpiscine_device.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class RPCDeviceFactory : public DeviceFactory {
public:
    Status CreateDevices(const SessionOptions &options, const string &name_prefix,
                         std::vector<Device *> *devices) override {
        LOG(INFO) << "RPCDeviceFactory got configproto: " << options.config.DebugString();

        int n = 1;
        auto iter = options.config.device_count().find("RPC");
        if (iter != options.config.device_count().end()) {
            n = iter->second;
        }

        for (int i = 0; i < n; i++) {
            static ZmqRpcClient rpc(options.env, "tcp://localhost:5501");

            string name = strings::StrCat(name_prefix, "/device:RPC:", i);
            LOG(INFO) << "Creating RPC device: " << name;

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
