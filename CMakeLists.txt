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

message(STATUS "XLA_SOURCES=${XLA_SOURCES}")
#message(STATUS "XLA_HEADERS=${XLA_HEADERS}")

if(NOT TORCH_PATH)
    get_filename_component(TORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
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
include_directories(${TORCH_PATH}/torch/include/TH)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fwrapv ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-sign-compare ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-return-type ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-macro-redefined ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-return-std-move ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-DTORCH_API_INCLUDE_EXTENSION_H ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-DTORCH_EXTENSION_NAME=_XLAC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")

add_library(ptxla SHARED ${XLA_SOURCES} ${XLA_HEADERS})

add_dependencies(ptxla link_torch link_torch_python link_c10 link_shm link_torch_global_deps)

set_target_properties(
        ptxla
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib.linux-x86_64-3.6/
)

message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

set(LIBXLA_COMPUTATION_CLIENT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib/libxla_computation_client.so")
message(STATUS "LIBXLA_COMPUTATION_CLIENT_LIB: ${LIBXLA_COMPUTATION_CLIENT_LIB}")
target_link_libraries(ptxla "${LIBXLA_COMPUTATION_CLIENT_LIB}")

target_link_libraries(
        ptxla
        -Wl,--unresolved-symbols=ignore-in-shared-libs
        "${TORCH_LIBRARIES}"
        "${PTPY_LIB}"
        "${PYTHON_LIBRARY}"
        -lutil
        -pthread
        -lstdc++
        -ldl
)

add_dependencies(ptxla libxla_computation_client)

add_custom_target(
        xlac ALL
        COMMAND "ln" "-s" "-f" "$<TARGET_FILE:ptxla>" "${CMAKE_CURRENT_LIST_DIR}/_XLAC.cpython-36m-x86_64-linux-gnu.so"
        DEPENDS ptxla
)

if (BUILD_TENSORFLOW)
    add_dependencies(ptxla libxla_computation_client)
endif()
