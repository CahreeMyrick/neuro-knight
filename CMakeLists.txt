cmake_minimum_required(VERSION 3.16)

project(neuro-knight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers from src/
include_directories(${CMAKE_SOURCE_DIR}/src)

# Engine source files
file(GLOB_RECURSE ENGINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Self-play executable
add_executable(t1
    tests/t1.cpp
    ${ENGINE_SOURCES}
)

target_include_directories(t1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)


target_compile_options(t1 PRIVATE
    -Wall
    -Wextra
    -pedantic
)
