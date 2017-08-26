export TF_CPP_MIN_VLOG_LEVEL=3
export TF_CPP_MIN_LOG_LEVEL=
export CUDA_VISIBLE_DEVICES=
export PYTHON_BIN_PATH="/gpfs/gpfs0/groups/chowdhury/peifeng/.local/venvs/tfbuild/bin/python"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CC_OPT_FLAGS="-mcpu=power8"
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=1
export TF_NEED_RPC=1
export GCC_HOST_COMPILER_PATH="/gpfs/gpfs0/software/rhel72/packages/gcc/5.4.0/bin/gcc"
export TF_CUDA_VERSION="8.0"
export CUDA_TOOLKIT_PATH="/gpfs/gpfs0/software/rhel72/packages/cuda/8.0"
export TF_CUDNN_VERSION="5"
export CUDNN_INSTALL_PATH="$CUDA_TOOLKIT_PATH"
export TF_CUDA_COMPUTE_CAPABILITIES="3.7,6.0"

module load compilers/gcc/5.4.0

function repatch {
    git apply tools/path-zeromq.patch
    git apply tools/path-gcc54.patch
    git apply tools/debug-build.patch
}

function bb() {
    bazel build -c opt --config=cuda "$@" //tensorflow:libtensorflow.so //tensorflow:libtensorflow_kernels.so //tensorflow/tools/pip_package:build_pip_package
}

BUILD_BRANCH=tfbuild
function cf() {
    cur_br=$(git rev-parse --abbrev-ref HEAD)
    git checkout -b $BUILD_BRANCH

    ./configure "$@"

    git checkout $cur_br
    git branch -D $BUILD_BRANCH
}

function bbi() {
    cur_br=$(git rev-parse --abbrev-ref HEAD)
    git checkout -b $BUILD_BRANCH

    bb "$@" && bazel-bin/tensorflow/tools/pip_package/build_pip_package $HOME/downloads && pip uninstall -y tensorflow && pip install $HOME/downloads/*.whl

    git checkout $cur_br
    git branch -D $BUILD_BRANCH
}

#build && bazel-bin/tensorflow/tools/pip_package/build_pip_package $HOME/downloads && pip uninstall -y tensorflow && pip install $HOME/downloads/*.whl
