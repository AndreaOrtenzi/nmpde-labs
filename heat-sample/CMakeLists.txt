cmake_minimum_required(VERSION 3.12.0)
project(labHeatSample LANGUAGES CXX C)

include(../common/cmake-common.cmake)


add_executable(main.exe src/main.cpp src/ParametricHeat.cpp)
deal_ii_setup_target(main.exe)
