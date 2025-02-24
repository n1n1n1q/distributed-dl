cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(distributed_dl)

# copied from docs ---------------------------------------------

message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app pytorch_playground/example_app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${MPI_LIBRARIES}" ${CMAKE_PREFIX_PATH}/lib/libc10d.a)

if (MPI_COMPILE_FLAGS)
    set_target_properties(example-app PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif ()

if (MPI_LINK_FLAGS)
    set_target_properties(example-app PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif ()

set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# copied from docs ---------------------------------------------

# --------------------------------------------------------------
add_executable(distributed_dl src/main.cpp)
