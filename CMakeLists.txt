cmake_minimum_required(VERSION 3.16)
project(neuro-knight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Engine source files
file(GLOB_RECURSE ENGINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Engine library
add_library(engine STATIC
    ${ENGINE_SOURCES}
)

target_include_directories(engine PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_options(engine PRIVATE
    -Wall
    -Wextra
    -pedantic
)

# Self-play / test executable
add_executable(t1
    tests/t1.cpp
)

target_link_libraries(t1 PRIVATE engine)

# Web match JSON runner
add_executable(run_web_match
    scripts/run_web_match.cpp
)

target_link_libraries(run_web_match PRIVATE engine)
