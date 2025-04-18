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


add_executable(resnet src/examples/resnet.cpp src/serializedtensor.hpp src/distributed.hpp)
target_link_libraries(resnet
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)


set(TEST_EXECUTABLES gather scatter basic-communication reduce allreduce)

# ------------------------------------------------------------------------

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

# Error handling
if (NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR "NCCL headers (nccl.h) not found. Set -DNCCL_ROOT=/path/to/nccl.")
endif ()
if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL library (libnccl.so) not found. Set -DNCCL_ROOT=/path/to/nccl.")
endif ()

add_executable(nccl-test src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp src/examples/nccl_test.cpp)
target_include_directories(nccl-test PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(nccl-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_Library}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)

add_executable(resnet-cuda src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp src/examples/resnet-cuda.cpp)
target_include_directories(resnet-cuda PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(resnet-cuda
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_Library}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)


add_executable(resnet-unified src/distributed.hpp src/nccl_distr.cuh src/serializedtensor.hpp src/examples/resnet-with-model-trainer.cpp)
target_include_directories(resnet-unified PRIVATE ${NCCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(resnet-unified
        PRIVATE
        "${TORCH_LIBRARIES}"
        "${NCCL_Library}"
        CUDA::cudart
        Boost::mpi
        Boost::serialization
)

foreach (EXEC_NAME IN LISTS TEST_EXECUTABLES)
    string(REPLACE "-" "_" SOURCE_BASE "${EXEC_NAME}")
    set(SOURCE_FILE "${SOURCE_BASE}.cpp")

    add_executable(${EXEC_NAME} src/examples/${SOURCE_FILE})

    target_link_libraries(${EXEC_NAME}
            PRIVATE
            "${TORCH_LIBRARIES}"
            Boost::mpi
            Boost::serialization
            MPI::MPI_CXX
    )

endforeach ()