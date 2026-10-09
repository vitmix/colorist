cmake_minimum_required(VERSION 3.24...4.4)
project(colorist LANGUAGES CXX)

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.deps/opencv")
find_package(OpenCV 4.14 REQUIRED COMPONENTS core imgcodecs)

add_executable(colorist sources/main.cpp)
set_target_properties(colorist PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(colorist PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(colorist PRIVATE ${OpenCV_LIBS})
if(MINGW)
    target_link_options(colorist PRIVATE -static)
endif()
