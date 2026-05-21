# Build the CAG MLIR PassInstrumentation as a shared library that can be
# loaded by Triton via its CUDABackend.instrumentation hook.
#
# Required upstream packages:
#   * LLVM + MLIR development headers/libraries matching Triton's pinned LLVM
#     (Triton 3.6 currently pins ~LLVM 20.x; check `triton/cmake/llvm-hash.txt`
#     in the Triton source tree for the exact commit).
#   * A C++17 toolchain (gcc >= 9 or clang >= 10).
#
# Build:
#   cmake -S cpp -B cpp/build \
#       -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
#       -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
#       -DCMAKE_BUILD_TYPE=Release
#   cmake --build cpp/build -j
#
# Output:
#   cpp/build/libCagPassInstrumentation.so

cmake_minimum_required(VERSION 3.20)
project(CagPassInstrumentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find LLVM and MLIR. Both must come from the same build tree as Triton's
# pinned LLVM, otherwise ABI mismatches will cause silent corruption at
# runtime (LLVM/MLIR do not version their C++ ABI).
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(CagPassInstrumentation SHARED
  CagPassInstrumentation.cpp
)

target_include_directories(CagPassInstrumentation
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against MLIR core libs. Pass/IR are required; Support brings in
# llvm::raw_ostream. We deliberately do NOT link Triton's own libs --
# this library is dialect-agnostic and works against any MLIR PassManager.
target_link_libraries(CagPassInstrumentation
  PRIVATE
    MLIRPass
    MLIRIR
    MLIRSupport
)

set_target_properties(CagPassInstrumentation PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Export only the C entry point so it's stable across LLVM versions.
if(UNIX AND NOT APPLE)
  set(VERSION_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/CagPassInstrumentation.version")
  file(WRITE "${VERSION_SCRIPT}"
"{
  global:
    cag_attach_to_pass_manager;
  local:
    *;
};
")
  target_link_options(CagPassInstrumentation PRIVATE
    "LINKER:--version-script=${VERSION_SCRIPT}")
endif()
