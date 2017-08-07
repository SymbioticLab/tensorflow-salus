licenses(["notice"])

cc_library(
    name = "zmq_cpp",
    srcs = ["lib/libzmq.so"],
    hdrs = [
        "include/zmq.h",
        "include/zmq.hpp",
        "include/zmq_utils.h",
    ],
    strip_include_prefix = "include/",
    copts = [
        "-fexceptions",
    ],
    # this is needed to make bazel add the path using -isystem.
    # Bazel defaults to add path using -iquote, thus #include<...> will not find it.
    includes = ["."],
    visibility = ["//visibility:public"],
)
