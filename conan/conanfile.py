from conans import ConanFile, tools
import os


class TensorflowsalusConan(ConanFile):
    name = "tensorflow-devel"
    # we remove compiler from settings so that we can use the package from different compilers
    settings = "os", "arch"
    description = "Development package for tensorflow. This is a total hack exposing all internals of TensorFlow"
    url = "https://github.com/SymbioticLab/tensorflow-salus"
    license = "Apache-2.0"
    author = "Peifeng Yu"

    default_user = "symbioticlab"
    default_channel = "testing"

    export_sources = "TensorFlowConfig.cmake"

    # comma-separated list of requirements
    requires = "zeromq/4.3.2@symbioticlab/stable", "cppzmq/4.6.0@symbioticlab/stable"
    default_options = {
        "zeromq:shared": True,
        "libsodium:shared": True,
    }

    def imports(self):
        # import dependency packages to a central localtion for bazel to consume
        self.copy("*.*", "include", "include")
        self.copy("*.so*", "lib", "lib")

    def package(self):
        # Package header files
        self.copy(
            "*.h", "include", "bazel-tensorflow",
            excludes=(
                "_bin/*", "bazel-out/*",
                # eigen will be copied in whole
                "third_party/eigen3/*",
                "external/eigen_archive/*"
            )
        )
        self.copy("*.h", "include", "bazel-genfiles")

        # Package proto files
        self.copy(
            "*.proto", "include", "bazel-tensorflow",
            excludes=(
                "_bin/*", "bazel-out/*",
            )
        )
        self.copy("*.proto", "include", "bazel-genfiles")

        # Package whole Eigen library, which can't be cover in header files
        self.copy("*", "include/third_party/eigen3", "bazel-tensorflow/third_party/eigen3")
        self.copy("*", "include/external/eigen_archive", "bazel-tensorflow/external/eigen_archive")

        # Package so files
        for libname in ["tensorflow_kernels"]:
            lib = self._sonamehelper(libname)
            self.copy(lib.barename, "lib", "bazel-bin/tensorflow", keep_path=False, symlinks=True)
            self.copy(lib.soname, "lib", "bazel-bin/tensorflow", keep_path=False, symlinks=True)
            self.copy(lib.fullname, "lib", "bazel-bin/tensorflow", keep_path=False)
            self.copy(lib.params_file, "lib", "bazel-bin/tensorflow", keep_path=False)

        # Ship a cmake module, conan can automatically find this
        self.copy("TensorFlowConfig.cmake", "lib/cmake/tensorflow", keep_path=False)

    def _sonamehelper(self, libname):
        """Fix soname"""
        class soname(object):
            def __init__(self, libname, version):
                self.libname = libname
                self._ver = version
                self._sover = version.split('.')[0]
            @property
            def barename(self):
                return "lib{}.so".format(self.libname)
            @property
            def soname(self):
                return self.barename + "." + self._sover
            @property
            def fullname(self):
                return self.barename + "." + self._ver
            @property
            def params_file(self):
                return self.barename + "-2.params"
            @property
            def full_params_file(self):
                return self.fullname + "-2.params"

        lib = soname(libname, self.version.split('-')[0])
        with tools.chdir(os.path.join(self.build_folder, "bazel-bin/tensorflow")):
            self.run("ln -sf {} {}".format(lib.fullname, lib.barename))
            self.run("ln -sf {} {}".format(lib.fullname, lib.soname))
            self.run("cp {} {}".format(lib.full_params_file, lib.params_file))
        return lib

    def package_info(self):
        self.cpp_info.name = self.name
