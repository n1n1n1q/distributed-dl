cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl)

# set(CUDACXX "/usr/local/cuda-12.8/bin/nvcc")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
# set(CMAKE_PREFIX_PATH /home/oleh/.libs/libtorch/share/cmake/Torch)
# set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Torch REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS mpi serialization)
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

### Include Directories ###
include_directories(${PROJECT_SOURCE_DIR}/src)

# Common test executable names
set(TEST_EXECUTABLES gather scatter basic-communication reduce allreduce)

# MPI applications
add_executable(resnet src/examples/mpi/resnet.cpp src/serializedtensor.hpp src/distributed.hpp)
target_link_libraries(resnet
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)

# MPI test executables
foreach(EXEC_NAME IN LISTS TEST_EXECUTABLES)
    string(REPLACE "-" "_" SOURCE_BASE "${EXEC_NAME}")
    set(SOURCE_FILE "${SOURCE_BASE}.cpp")

    add_executable(${EXEC_NAME} src/examples/mpi/${SOURCE_FILE})

    target_link_libraries(${EXEC_NAME}
            PRIVATE
            "${TORCH_LIBRARIES}"
            Boost::mpi
            Boost::serialization
            MPI::MPI_CXX
    )
endforeach()

# ---------------------------------
# CUDA and NCCL configuration

find_package(CUDAToolkit REQUIRED)

set(NCCL_ROOT "" CACHE PATH "Path to NCCL installation directory")
find_path(NCCL_INCLUDE_DIR
        nccl.h
        HINTS ${NCCL_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS}
        PATHS /usr/include /usr/local/include
)
find_library(NCCL_LIBRARY
        nccl
        HINTS ${NCCL_ROOT}/lib ${NCCL_ROOT}/lib64 ${CUDAToolkit_LIBRARY_DIR}
        PATHS /usr/lib /usr/lib64 /usr/local/lib
)

if (NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR "NCCL headers (nccl.h) not found. Set -DNCCL_ROOT=/path/to/nccl.")
endif ()
if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL library (libnccl.so) not found. Set -DNCCL_ROOT=/path/to/nccl.")
endif ()

# NCCL test executables
foreach(EXEC_NAME IN LISTS TEST_EXECUTABLES)
    string(REPLACE "-" "_" SOURCE_BASE "${EXEC_NAME}")
    set(SOURCE_FILE "${SOURCE_BASE}.cpp")

    add_executable(${EXEC_NAME}_gpu src/examples/nccl/${SOURCE_FILE})
    target_include_directories(${EXEC_NAME}_gpu PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${EXEC_NAME}_gpu
            PRIVATE
            "${TORCH_LIBRARIES}"
            Boost::mpi
            Boost::serialization
            MPI::MPI_CXX
            "${NCCL_LIBRARY}"
            CUDA::cudart
    )
endforeach()

# Additional CUDA applications
add_executable(nccl-test src/examples/nccl_test.cpp src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp)
target_include_directories(nccl-test PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nccl-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_LIBRARY}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)

add_executable(resnet-cuda src/examples/resnet-cuda.cpp src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp)
target_include_directories(resnet-cuda PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(resnet-cuda
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_LIBRARY}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)

add_executable(resnet-unified src/examples/resnet-with-model-trainer.cpp src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp)
target_include_directories(resnet-unified PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(resnet-unified
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_LIBRARY}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)
