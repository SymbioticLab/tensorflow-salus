# Locates the tensorFlow library and include directories.
# Copyright (C) 2017 Peifeng Yu <peifeng@umich.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Usage Documentation
#
# Cache variables: (not for direct use in CMakeLists.txt)
#  TensorFlow_LIBRARY
#  TensorFlow_Kernel_LIBRARY
#  TensorFlow_INCLUDE_DIR
#
# Non-cache variables you might use in your CMakeLists.txt:
#  TensorFlow_FOUND
#
#  TensorFlow_LIBRARIES - Libraries to link for consumer targets
#  TensorFlow_INCLUDE_DIRS - Include directories
#  TensorFlow_PROTO_DIRS - Include directories for protobuf imports
#
# Adds the following targets:
#  tensorflow::headers - A header only interface library
#  tensorflow::framework - The whole tensorflow library
#  tensorflow::kernels - All kernels
#
# Use this module this way:
#  find_package(TensorFlow)
#  add_executable(myapp ${SOURCES})
#  target_link_libraries(myapp tensorflow::framework)
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (CMake standard module)

# assume the file is in lib/cmake/tensorflow/TensorFlowConfig.cmake
get_filename_component(TensorFlow_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

find_path(TensorFlow_INCLUDE_DIR
    NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        external
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

find_library(TensorFlow_LIBRARY
    NAMES
        tensorflow_framework
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

find_library(TensorFlow_Kernel_LIBRARY
    NAMES
        tensorflow_kernels
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY TensorFlow_Kernel_LIBRARY)

# set TensorFlow_FOUND
include(FindPackageHandleStandardArgs)
unset(TensorFlow_FOUND)
find_package_handle_standard_args(TensorFlow DEFAULT_MSG
    TensorFlow_INCLUDE_DIR
    TensorFlow_LIBRARY
    TensorFlow_Kernel_LIBRARY
)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    message("-- Found TensorFlow: ${TensorFlow_ROOT}")
    set(tf_binary_path ${TensorFlow_ROOT}/lib)

    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY} ${TensorFlow_Kernel_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS
        ${TensorFlow_INCLUDE_DIR}
        ${TensorFlow_INCLUDE_DIR}/external/eigen_archive
        ${TensorFlow_INCLUDE_DIR}/external/nsync/public
    )
    # This is the same as the include dir
    set(TensorFlow_PROTO_DIRS ${TensorFlow_INCLUDE_DIR})

    # locate cuda in tensorflow
    if(EXISTS ${TensorFlow_INCLUDE_DIR}/external/local_config_cuda/cuda/cuda/cuda_config.h)
        file(STRINGS ${TensorFlow_INCLUDE_DIR}/external/local_config_cuda/cuda/cuda/cuda_config.h
             tf_cuda_config REGEX "TF_CUDA_TOOLKIT_PATH")
        string(REGEX REPLACE "^#define TF_CUDA_TOOLKIT_PATH \"(.+)\"$" "\\1" tf_cuda_path ${tf_cuda_config})
        message("-- Found TF CUDA: ${tf_cuda_path}")

        set(tf_cuda_link_path_flag -L${tf_cuda_path}/lib64)
        list(APPEND TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR}/external/local_config_cuda/cuda/)
    else()
        message("-- Found TF CUDA: NotFound")
        set(tf_cuda_link_path_flag "")
    endif()

    # Add imported targets
    add_library(tensorflow::headers INTERFACE IMPORTED)
    set_property(TARGET tensorflow::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${TensorFlow_INCLUDE_DIRS}
    )

    add_library(tensorflow::framework SHARED IMPORTED)
    file(STRINGS ${tf_binary_path}/libtensorflow_framework.so-2.params FrameworkLinkLibraries REGEX "^-l")
    set_property(TARGET tensorflow::framework PROPERTY INTERFACE_LINK_LIBRARIES
        tensorflow::headers
        ${tf_cuda_link_path_flag}
        ${FrameworkLinkLibraries}
    )
    set_property(TARGET tensorflow::framework PROPERTY IMPORTED_LOCATION ${TensorFlow_LIBRARY})

    add_library(tensorflow::kernels SHARED IMPORTED)
    file(STRINGS ${tf_binary_path}/libtensorflow_kernels.so-2.params KernelLibraries REGEX "^-l")
    set_property(TARGET tensorflow::kernels PROPERTY INTERFACE_LINK_LIBRARIES
        tensorflow::headers
        ${tf_cuda_link_path_flag}
        ${KernelLinkLibraries}
    )
    set_property(TARGET tensorflow::kernels PROPERTY IMPORTED_LOCATION ${TensorFlow_Kernel_LIBRARY})

endif()
