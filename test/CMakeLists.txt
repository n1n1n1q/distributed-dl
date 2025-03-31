cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl_tests)


set(Torch_DIR "../libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

### find packages ###

find_package(Boost 1.70 REQUIRED COMPONENTS mpi serialization)
find_package(MPI REQUIRED)

### tests ###

set(TEST_EXECUTABLES gather-test scatter-test igather-test)

foreach(EXEC_NAME IN LISTS TEST_EXECUTABLES)
    string(REPLACE "-" "_" SOURCE_BASE "${EXEC_NAME}")
    set(SOURCE_FILE "${SOURCE_BASE}.cpp")
    
    add_executable(${EXEC_NAME} ${SOURCE_FILE})
    
    target_include_directories(${EXEC_NAME} PRIVATE ../src)
    
    target_link_libraries(${EXEC_NAME}
        PRIVATE
        "${TORCH_LIBRARIES}"
        Boost::mpi
        Boost::serialization
        MPI::MPI_CXX
    )
    
    set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 17)
endforeach()


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