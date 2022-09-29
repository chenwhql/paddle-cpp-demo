cmake_minimum_required(VERSION 3.10)
project(paddle_cpp_demo CXX)

message(STATUS "PYTHON_LIBRARY = ${PYTHON_LIBRARY}")

set(include_dirs ${PYTHON_LIBRARY}/paddle/include/)
set(link_libraries ${PYTHON_LIBRARY}/paddle/fluid/libpaddle.so)
set(link_libraries_path ${CMAKE_SOURCE_DIR};${PYTHON_LIBRARY}/paddle/libs;${PYTHON_LIBRARY}/paddle/fluid/)

include_directories(${include_dirs})
link_directories(${link_libraries_path})

set(name main)
set(source main.cc)
add_executable(${name} ${source})
target_link_libraries(${name} ${link_libraries})
