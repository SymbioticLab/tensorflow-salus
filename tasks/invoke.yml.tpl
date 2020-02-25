buildcfg:
  bazelArgs:
  env:
    TF_CPP_MIN_VLOG_LEVEL: 3
    TF_CPP_MIN_LOG_LEVEL:
    CUDA_VISIBLE_DEVICES:

    PYTHON_BIN_PATH: ${PYTHON_BIN_PATH}
    USE_DEFAULT_PYTHON_LIB_PATH: 1

    CC_OPT_FLAGS: -march=native

    ZEROMQ_PATH: ${ZEROMQ_PATH}

    TF_NEED_CUDA: 1
    TF_CUDA_VERSION: ${CUDA_VERSION}
    CUDA_TOOLKIT_PATH: ${CUDA_PATH}
    TF_CUDNN_VERSION: ${CUDNN_VERSION}
    CUDNN_INSTALL_PATH: ${CUDA_PATH}
    TF_CUDA_COMPUTE_CAPABILITIES: 6.1
    TF_CUDA_CLANG: 0
    GCC_HOST_COMPILER_PATH: ${GCC_BIN_PATH}
