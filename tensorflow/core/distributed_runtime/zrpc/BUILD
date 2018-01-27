# Description:
#   ZRPC communication interfaces and implementations for TensorFlow.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load(
    "//tensorflow:tensorflow.bzl",

    "tf_cc_test",
    "tf_cuda_library",
    "tf_cuda_cc_test",
    "tf_cuda_cc_tests",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
    "tf_proto_library_cc",
)
load(
    "//tensorflow/core:platform/default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# For package specific build config
load(
    ":zrpc.bzl",
    "zrpc_copts",
)

package(default_visibility = [
    "//tensorflow:internal",
])

cc_library(
    name = "zrpc_runtime",
    visibility = ["//visibility:public"],
    deps = [
        ":zrpc_session",
        ":zrpc_exechelper",
    ],
    copts = zrpc_copts(),
)

tf_proto_library_cc(
    name = "executor_protos",
    srcs = [
        "protos/executor.proto",
        "protos/tfoplibrary.proto",
    ],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
)

cc_library(
    name = "zrpc_util",
    hdrs = ["zrpc_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "@local_config_zeromq//:zmq_cpp",
    ],
    copts = zrpc_copts(),
)

cc_library(
    name = "zrpc_session",
    srcs = ["zrpc_session.cc"],
    hdrs = ["zrpc_session.h"],
    deps = [
        ":zrpc_remote_master",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:master_interface",
    ],
    copts = zrpc_copts(),
    alwayslink = 1,
)

cc_library(
    name = "zrpc_remote_master",
    srcs = [
        "zrpc_remote_master.cc",
        "zrpc_master_service_stub.cc",
    ],
    hdrs = [
        "zrpc_remote_master.h",
        "zrpc_master_service_stub.h",
    ],
    deps = [
        ":zrpc_util",
        ":executor_protos_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime:master_interface",
        "@local_config_zeromq//:zmq_cpp",
    ],
    copts = zrpc_copts(),
    alwayslink = 1,
)

cc_library(
    name = "zrpc_exechelper",
    srcs = [
        "exechelper/tfoplibraryproxy.cc",
        "exechelper/graphview.cc",
        "exechelper/mdgraphmgr.cc",
        "exechelper/devicefactories.cc",
        "exechelper/allocators.cc",
        "exechelper/memorytypes.cc",
        "exechelper/paginghelper.cc",
    ],
    hdrs = [
        "exechelper/tfoplibraryproxy.h",
        "exechelper/graphview.h",
        "exechelper/mdgraphmgr.h",
        "exechelper/devicefactories.h",
        "exechelper/allocators.h",
        "exechelper/memorytypes.h",
        "exechelper/paginghelper.h",
    ],
    deps = [
        ":zrpc_util",
        ":zrpc_worker_cache",
        ":zrpc_worker_service",
        ":zrpc_rendezvous_mgr",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
    copts = zrpc_copts(),
    alwayslink = 1,
)

cc_library(
   name = "zrpc_worker_cache",
   srcs = ["zrpc_worker_cache.cc"],
   hdrs = ["zrpc_worker_cache.h"],
   deps = [
       "//tensorflow/core/distributed_runtime:worker_cache",
       "//tensorflow/core/distributed_runtime:worker_cache_logger",
       "//tensorflow/core/distributed_runtime:worker_cache_partial",
       "//tensorflow/core/distributed_runtime:worker_interface",
   ],
    copts = zrpc_copts(),
)

tf_cuda_library(
   name = "zrpc_worker_service",
   srcs = ["zrpc_worker_service.cc"],
   hdrs = ["zrpc_worker_service.h"],
   deps = [
       ":zrpc_tensor_coding",
       "//tensorflow/core:core_cpu_internal",
       "//tensorflow/core:framework",
       "//tensorflow/core:gpu_runtime",
       "//tensorflow/core:lib",
       "//tensorflow/core:lib_internal",
       "//tensorflow/core:worker_proto_cc",
       "//tensorflow/core/distributed_runtime:graph_mgr",
       "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
       "//tensorflow/core/distributed_runtime:worker",
       "//tensorflow/core/distributed_runtime:worker_cache",
       "//tensorflow/core/distributed_runtime:worker_env",
   ],
    copts = zrpc_copts(),
)

cc_library(
    name = "zrpc_tensor_coding",
    srcs = ["zrpc_tensor_coding.cc"],
    hdrs = ["zrpc_tensor_coding.h"],
    deps = [
        ":zrpc_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:worker_proto_cc",
        "@local_config_zeromq//:zmq_cpp",
    ],
    copts = zrpc_copts(),
)

cc_library(
    name = "zrpc_rendezvous_mgr",
    srcs = [
        "zrpc_rendezvous_mgr.cc",
        "zrpc_wrapped_devicecontext.cc",
    ],
    hdrs = [
        "zrpc_rendezvous_mgr.h",
        "zrpc_wrapped_devicecontext.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
        "//tensorflow/core/distributed_runtime:tensor_coding",
    ],
    copts = zrpc_copts(),
)

#cc_library(
#    name = "grpc_remote_worker",
#    srcs = ["grpc_remote_worker.cc"],
#    hdrs = ["grpc_remote_worker.h"],
#    deps = [
#        ":grpc_client_cq_tag",
#        ":grpc_util",
#        ":grpc_worker_service_impl",
#        "//tensorflow/core:core_cpu_internal",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:lib_internal",
#        "//tensorflow/core:worker_proto_cc",
#        "//tensorflow/core/distributed_runtime:worker_cache_logger",
#        "//tensorflow/core/distributed_runtime:worker_interface",
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#cc_library(
#    name = "grpc_worker_service_impl",
#    srcs = ["grpc_worker_service_impl.cc"],
#    hdrs = ["grpc_worker_service_impl.h"],
#    deps = [
#        ":grpc_serialization_traits",
#        "//tensorflow/core:worker_proto_cc",
#        "//tensorflow/core/distributed_runtime:worker_interface",
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#cc_library(
#    name = "grpc_serialization_traits",
#    srcs = [],
#    hdrs = ["grpc_serialization_traits.h"],
#    deps = [
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#
#
#cc_binary(
#    name = "grpc_tensorflow_server",
#    srcs = [
#        "grpc_tensorflow_server.cc",
#    ],
#    deps = [
#        ":grpc_server_lib",
#        "//tensorflow/core:all_kernels",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:framework_internal",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core/distributed_runtime:server_lib",
#        "//tensorflow/core/kernels:data_flow",
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#tf_cuda_library(
#    name = "grpc_testlib_ops",
#    testonly = 1,
#    srcs = ["grpc_testlib_ops.cc"],
#    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
#    deps = [
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#    ],
#    alwayslink = 1,
#)
#
#cc_binary(
#    name = "grpc_testlib_server",
#    testonly = 1,
#    srcs = [
#        "grpc_testlib_server.cc",
#    ],
#    deps = [
#        ":grpc_server_lib",
#        ":grpc_testlib_ops",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:framework_internal",
#        "//tensorflow/core:lib",
#        "//tensorflow/core/distributed_runtime:server_lib",
#        "//tensorflow/core/kernels:constant_op",
#        "//tensorflow/core/kernels:cwise_op",
#        "//tensorflow/core/kernels:dense_update_ops",
#        "//tensorflow/core/kernels:identity_op",
#        "//tensorflow/core/kernels:matmul_op",
#        "//tensorflow/core/kernels:reduction_ops",
#        "//tensorflow/core/kernels:variable_ops",
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#tf_cuda_library(
#    name = "grpc_testlib",
#    testonly = 1,
#    srcs = ["grpc_testlib.cc"],
#    hdrs = ["grpc_testlib.h"],
#    data = [
#        ":grpc_testlib_server",
#    ],
#    deps = [
#        ":grpc_session",
#        ":grpc_testlib_ops",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core:test",
#    ],
#    alwayslink = 1,
#)
#
#
#tf_cuda_cc_tests(
#    name = "rpc_tests",
#    size = "small",
#    srcs = [
#        "grpc_channel_test.cc",
#        "rpc_rendezvous_mgr_test.cc",
#    ],
#    linkopts = select({
#        "//tensorflow:darwin": ["-headerpad_max_install_names"],
#        "//conditions:default": [],
#    }),
#    linkstatic = tf_kernel_tests_linkstatic(),
#    tags = tf_cuda_tests_tags() + [],
#    deps = [
#        ":grpc_channel",
#        ":grpc_server_lib",
#        ":grpc_session",
#        ":grpc_testlib",
#        ":rpc_rendezvous_mgr",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:core_cpu_internal",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:master_proto_cc",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core:test",
#        "//tensorflow/core:test_main",
#        "//tensorflow/core:testlib",
#        "//tensorflow/core/distributed_runtime:server_lib",
#    ],
#)
#
#tf_cc_test(
#    name = "grpc_tensor_coding_test",
#    size = "small",
#    srcs = ["grpc_tensor_coding_test.cc"],
#    deps = [
#        ":grpc_tensor_coding",
#        ":grpc_testlib",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:core_cpu_internal",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core:test",
#        "//tensorflow/core:test_main",
#        "//tensorflow/core:testlib",
#        "//tensorflow/core:worker_proto_cc",
#        "@grpc//:grpc++_unsecure",
#    ],
#)
#
#tf_cuda_cc_test(
#    name = "grpc_session_test",
#    size = "medium",
#    srcs = ["grpc_session_test.cc"],
#    linkstatic = tf_kernel_tests_linkstatic(),
#    tags = tf_cuda_tests_tags() + ["manual"],
#    deps = [
#        ":grpc_channel",
#        ":grpc_server_lib",
#        ":grpc_session",
#        ":grpc_testlib",
#        ":rpc_rendezvous_mgr",
#        "//tensorflow/core:core_cpu",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:master_proto_cc",
#        "//tensorflow/core:protos_all_cc",
#        "//tensorflow/core:test",
#        "//tensorflow/core:test_main",
#        "//tensorflow/core:testlib",
#        "//tensorflow/core/distributed_runtime:server_lib",
#        "//tensorflow/core/kernels:constant_op",
#        "//tensorflow/core/kernels:dense_update_ops",
#        "//tensorflow/core/kernels:matmul_op",
#        "//tensorflow/core/kernels:variable_ops",
#    ],
#)
#
