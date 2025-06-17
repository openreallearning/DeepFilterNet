cmake_minimum_required(VERSION 3.10)
project(deepfilter_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(../libsndfile libsndfile)

add_executable(enhance enhance.cpp)
add_executable(test_voice test_voice.cpp)
target_sources(test_voice PRIVATE enhance.cpp)
target_compile_definitions(test_voice PRIVATE DF_NO_MAIN)

# Include directories from pkg-config and current dir for header
target_include_directories(enhance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../libsndfile/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/include)
target_link_directories(enhance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/build/Linux/Debug/)
target_include_directories(test_voice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../libsndfile/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/include)
target_link_directories(test_voice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/build/Linux/Debug/)

# Link libraries
target_link_libraries(enhance PRIVATE sndfile onnxruntime)
target_link_libraries(test_voice PRIVATE sndfile onnxruntime)
