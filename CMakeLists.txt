cmake_minimum_required(VERSION 3.20)

project(
    CUDA_Matrix_Multiplication_Optimizer
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
    cpu_matmul
    src/cpu_matmul.cpp
)

target_compile_options(
    cpu_matmul
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

include(CTest)

if(BUILD_TESTING)
    add_executable(
        test_cpu_matmul
        tests/test_cpu_matmul.cpp
        src/cpu_matmul.cpp
    )

    target_compile_definitions(
        test_cpu_matmul
        PRIVATE MATMUL_TESTING
    )

    target_compile_options(
        test_cpu_matmul
        PRIVATE -Wall -Wextra -Wpedantic
    )

    add_test(
        NAME cpu_correctness
        COMMAND test_cpu_matmul
    )
endif()

if(BUILD_TESTING)
    add_executable(test_cpu_reference
        tests/test_cpu_reference.cpp
        src/cpu_matmul.cpp
    )
    target_compile_definitions(test_cpu_reference PRIVATE MATMUL_TESTING)
    target_compile_options(test_cpu_reference PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME cpu_reference COMMAND test_cpu_reference)
endif()
