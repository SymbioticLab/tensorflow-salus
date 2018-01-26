import os

class venv(object):
    def __init__(self, vdir):
        self.dir = vdir
        for name in ['pip', 'python']:
            setattr(self, name, os.path.join(self.dir, 'bin', name))

VENV = venv('/home/peifeng/.local/venvs/tfbuild')

CFGENV = {
    'TF_CPP_MIN_VLOG_LEVEL': '3',
    'TF_CPP_MIN_LOG_LEVEL': '',
    'CUDA_VISIBLE_DEVICES': '',

    'PYTHON_BIN_PATH': VENV.python,
    'USE_DEFAULT_PYTHON_LIB_PATH': '1',

    'CC_OPT_FLAGS': '-march=native',

    'TF_NEED_JEMALLOC': '1',
    'TF_NEED_GCP': '0',
    'TF_NEED_HDFS': '0',
    'TF_ENABLE_XLA': '0',
    'TF_NEED_OPENCL_SYCL': '0',
    'TF_NEED_MPI': '0',
    'TF_NEED_RPC': '1',

    'TF_NEED_CUDA': '1',
    'TF_CUDA_VERSION': '8.0',
    'CUDA_TOOLKIT_PATH': '/opt/cuda',
    'TF_CUDNN_VERSION': '6',
    'CUDNN_INSTALL_PATH': '/opt/cudnn6',
    'TF_CUDA_COMPUTE_CAPABILITIES': '6.1',
    'TF_CUDA_CLANG': '0',
    'GCC_HOST_COMPILER_PATH': '/usr/bin/gcc-5',

    'TF_SET_ANDROID_WORKSPACE': '0'
}

BUILD_BRANCH = 'tfbuild'

WORKSPACE = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))

try:
    from . import override
    orEnv = getattr(override, 'CFGENV')
    if orEnv:
        CFGENV.update(orEnv)
except:
    pass
