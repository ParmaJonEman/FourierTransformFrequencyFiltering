# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
find_package( OpenCV REQUIRED PATHS "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64")
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/bin")
project ("freqfilter")
add_executable (freqfilter "main.cpp" main.h)
target_link_libraries( freqfilter ${OpenCV_LIBS})