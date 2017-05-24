licenses(["notice"])

cc_library(
    name = "zmq_cpp",
    srcs = ["lib/libzmq.so"],
    hdrs = [
        "include/zmq.h",
        "include/zmq.hpp",
        "include/zmq_utils.h",
    ],
    copts = [
        "-fexceptions",
    ],
    visibility = ["//visibility:public"],
)
