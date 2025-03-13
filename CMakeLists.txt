cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl)


set(Torch_DIR "./libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# add_executable(example-app pytorch_playground/example_app.cpp)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")

### MPI ###
add_executable(mpi-test src/main.cpp src/mpi_wrapper/SerializedTensor.hpp src/distributed.hpp)

find_package(Boost 1.70 REQUIRED COMPONENTS mpi serialization)
find_package(MPI REQUIRED)
target_link_libraries(mpi-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)


set_property(TARGET mpi-test PROPERTY CXX_STANDARD 17)
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

# --------------------------------------------------------------
# add_executable(distributed_dl src/main.cpp)
