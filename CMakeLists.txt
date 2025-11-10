cmake_minimum_required(VERSION 3.12)
project(7.1_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

add_executable(matrix Hilbert/7.1_task.cpp)