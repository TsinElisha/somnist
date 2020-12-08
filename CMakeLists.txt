cmake_minimum_required(VERSION 3.17)
project(somnist)

set(CMAKE_CXX_STANDARD 14)

add_executable(somnist main.c apprentissage.c network.c initialise.c)
target_link_libraries(somnist dl m SDL)