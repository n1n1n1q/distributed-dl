cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl)


find_package(Torch REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS mpi serialization)
find_package(MPI REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# add_executable(example-app pytorch_playground/example_app.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")

### MPI ###
add_executable(mpi-test src/main.cpp src/serializedtensor.hpp src/distributed.hpp)

target_link_libraries(mpi-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)

add_executable(resnet pytorch-playground/resnet.cpp src/serializedtensor.hpp src/distributed.hpp)
target_link_libraries(resnet
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)


add_executable(nccl-test src/main.cpp)
target_link_libraries(nccl-test
        PRIVATE
        "${TORCH_LIBRARIES}"
)


set_property(TARGET mpi-test PROPERTY CXX_STANDARD 20)
set_property(TARGET resnet PROPERTY CXX_STANDARD 20)
set_property(TARGET nccl-test PROPERTY CXX_STANDARD 20)
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
# copied from docs ---------------------------------------------
