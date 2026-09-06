cmake_minimum_required(VERSION 3.10)

project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES
    src/*.cc
)

find_package(ZLIB REQUIRED)

add_library(cnpy
    external/cnpy/cnpy.cpp
)

target_include_directories(cnpy PUBLIC
    external/cnpy
)

target_link_libraries(cnpy PUBLIC
    ZLIB::ZLIB
)

add_executable(app
    ${SOURCES}
)

target_include_directories(app PRIVATE
    include
)

target_link_libraries(app PRIVATE
    cnpy
)

target_compile_options(app PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

target_link_options(app PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address>
)
