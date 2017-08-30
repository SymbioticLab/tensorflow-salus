# -*- Python -*-
# kate: syntax python

load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda",
    "if_mkl",
    "if_android_arm",
    "if_x86",
)

def zrpc_copts():
    return ([]
    + if_cuda(["-DGOOGLE_CUDA=1"])
    + if_mkl(["-DINTEL_MKL=1"])
    + if_android_arm(["-mfpu=neon"])
    + if_x86(["-msse3"])
    + select({
        "//tensorflow:android": [
            "-std=c++11",
            "-DTF_LEAN_BINARY",
            "-O2",
        ],
        "//tensorflow:darwin": [],
        "//tensorflow:windows": [
            "/DLANG_CXX11",
            "/D__VERSION__=\\\"MSVC\\\"",
            "/DPLATFORM_WINDOWS",
            "/DTF_COMPILE_LIBRARY",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
        ],
        "//tensorflow:ios": ["-std=c++11"],
        "//conditions:default": ["-pthread"]
    }))
