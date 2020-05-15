cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pytorch_xla CXX C)

#include(GNUInstallDirs)

message(STATUS "PATH=$ENV{PATH}")

include_directories(${PYTHON_INCLUDE_DIR})

message(STATUS "PYTHON_VERSION_STRING=${PYTHON_VERSION_STRING}")

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE KERNEL_NAME )
message( STATUS "Architecture: ${ARCHITECTURE}-${KERNEL_NAME}" )
set(CPYTHON_STR "cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}m-${ARCHITECTURE}-linux-gnu")
message(STATUS "CPYTHON_STR=${CPYTHON_STR}")

#if (NOT Torch_DIR)
#    set(Torch_DIR "${PYTORCH_DIR}/torch/share/cmake/Torch")
#endif()

#find_package(Torch REQUIRED)

file(GLOB
        XLA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/ops/*.cpp)

file(GLOB
        XLA_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/ops/*.h)

#message(STATUS "XLA_SOURCES=${XLA_SOURCES}")
#message(STATUS "XLA_HEADERS=${XLA_HEADERS}")

if(NOT TORCH_PATH)
    #get_filename_component(TORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    message(ERROR "TORCH_DIR needs to point to the pytorch source directory")
endif()
message(STATUS "TORCH_PATH: ${TORCH_PATH}")

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tensorflow)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-genfiles)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/protobuf_archive/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/com_google_protobuf/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/eigen_archive)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/com_google_absl)

include_directories(${TORCH_PATH})
include_directories(${TORCH_PATH}/torch/csrc)
include_directories(${TORCH_PATH}/torch/lib/tmp_install/include)
include_directories(${TORCH_PATH}/torch/include)
include_directories(SYSTEM ${TORCH_PATH}/torch/include/TH)

message(STATUS "ptxla CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fwrapv ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-sign-compare ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-return-type ${CMAKE_CXX_FLAGS}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wno-macro-redefined ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "-Wno-return-std-move ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "-DTORCH_API_INCLUDE_EXTENSION_H ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-DTORCH_EXTENSION_NAME=_XLAC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")

add_library(ptxla SHARED ${XLA_SOURCES} ${XLA_HEADERS})
add_dependencies(ptxla link_torch link_torch_python link_c10 link_shm link_torch_global_deps)

link_directories(${CMAKE_BINARY_DIR}/lib)

set_target_properties(
        ptxla
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib.linux-x86_64-3.7/
)

message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

target_link_libraries(
        ptxla
        -Wl,--unresolved-symbols=ignore-in-shared-libs
        "${TORCH_LIBRARIES}"
        "${PTPY_LIB}"
        "${PYTHON_LIBRARY}"
        -lutil
        -pthread
        -lstdc++
        -lprotobuf
        -ldl
)

add_custom_target(
        xlac ALL
        COMMAND "ln" "-s" "-f" "$<TARGET_FILE:ptxla>" "${CMAKE_CURRENT_LIST_DIR}/_XLAC.${CPYTHON_STR}.so"
        DEPENDS ptxla
)

set(CMAKE_EXE_LINKER_FLAGS_DEBUG:STRING="-Wl,--gdb-index" ${CMAKE_EXE_LINKER_FLAGS_DEBUG})

if (BUILD_TENSORFLOW)
    # get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
#    execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
#    execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')" OUTPUT_VARIABLE Tensorflow_LINK_FLAGS)
#    execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE Tensorflow_COMPILE_FLAGS)

#    message(STATUS "Tensorflow_COMPILE_FLAGS: ${Tensorflow_COMPILE_FLAGS}")
#    message(STATUS "Tensorflow_INCLUDE_DIRS: ${Tensorflow_INCLUDE_DIRS}")
#    message(STATUS "Tensorflow_LINK_FLAGS: ${Tensorflow_LINK_FLAGS}")

    #set(TF_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/third_party/tensorflow)

#    if (STANDALONE_COMPUTATION_CLIENT)
#        list(APPEND TENSORFLOW_TARGETS "//tensorflow/compiler/xla/xla_client:xla_computation_client_standalone.so")
#    else()
#        list(APPEND TENSORFLOW_TARGETS "//tensorflow/compiler/xla/xla_client:libxla_computation_client.so")
#    endif()

    #list(APPEND FRAMEWORK_LIBS ${TF_DIRECTORY}/bazel-out/host/bin/tensorflow/libtensorflow_framework.so)
        if (STANDALONE_COMPUTATION_CLIENT)
            set(LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME "_xla_computation_client_standalone.so")
            set(LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME lib${LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME})
            list(APPEND TENSORFLOW_TARGETS "//tensorflow/compiler/xla/xla_client:${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")
        else()
            list(APPEND FRAMEWORK_LIBS ${TF_DIRECTORY}/bazel-bin/tensorflow/python/_pywrap_tensorflow_internal.so)
            set(LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME "libxla_computation_client.so")
            set(LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME ${LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME})
            list(APPEND TENSORFLOW_TARGETS "//tensorflow/compiler/xla/xla_client:${LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME}" PARENT_SCOPE)
        endif()
        set(LIBXLA_COMPUTATION_CLIENT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")
        message(STATUS "LIBXLA_COMPUTATION_CLIENT_LIB: ${LIBXLA_COMPUTATION_CLIENT_LIB}")
        target_link_libraries(ptxla ${LIBXLA_COMPUTATION_CLIENT_LIB})
        target_link_libraries(ptxla ${FRAMEWORK_LIBS})
        add_dependencies(ptxla libxla_computation_client)
    #else()
        #message(STATUS "Not linking to libxla_computation_client")
    #endif()

    set(TF_SOURCE_FILE "${TF_DIRECTORY}/bazel-bin/tensorflow/compiler/xla/xla_client/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")
    set(TF_DEST_FILE "${CMAKE_CURRENT_LIST_DIR}/torch_xla/lib/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")

    message(STATUS "TO LINK: ${TF_SOURCE_FILE} -> ${TF_DEST_FILE}")

    add_custom_target(
            libxla_computation_client
            BYPRODUCTS "${LIBXLA_COMPUTATION_CLIENT_LIB}"
            COMMAND "ln" "-s" "-f"
            "${TF_SOURCE_FILE}"
            "${TF_DEST_FILE}"
            DEPENDS libxla_computation_client_build
    )

else()
    message(STATUS "Not building TensorFlow -- assuming the library exists: ${LIBXLA_COMPUTATION_CLIENT_LIB}")
endif()
