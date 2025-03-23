cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl_tests)


set(Torch_DIR "../libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

### find packages ###

find_package(Boost 1.70 REQUIRED COMPONENTS mpi serialization)
find_package(MPI REQUIRED)

### gather-test ###

add_executable(gather-test gather_test.cpp)

target_include_directories(gather-test PRIVATE ../src)

target_link_libraries(gather-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)
set_property(TARGET gather-test PROPERTY CXX_STANDARD 17)

### scatter-test ###

add_executable(scatter-test scatter_test.cpp)

target_include_directories(scatter-test PRIVATE ../src)

target_link_libraries(scatter-test
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
)
set_property(TARGET scatter-test PROPERTY CXX_STANDARD 17)


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