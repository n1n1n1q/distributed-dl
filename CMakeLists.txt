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

foreach(EXEC_NAME IN LISTS TEST_EXECUTABLES)
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

endforeach()