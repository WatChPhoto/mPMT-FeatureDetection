cmake_minimum_required(VERSION 3.2)
project(photogrammetry)

find_package(OpenCV REQUIRED)
find_package( ROOT 6.18 CONFIG REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++11 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
		    ${OpenCV_INCLUDE_DIRS}
		    ${ROOT_INCLUDE_DIRS})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hpp)

#ADD_LIBRARY( featurereco_lib ellipse.cpp hough_ellipse.cpp distance_to_ellipse.cpp PMTIdentified.cpp MedianTextReader.cpp ellipse_intersection.cpp featureFunctions.cpp)
add_executable(photogrammetry photogrammetry.cpp ${sources} ${headers})
target_link_libraries( photogrammetry ${OpenCV_LIBS} ${ROOT_LIBS} ROOT::Core ROOT::Hist ROOT::Graf)
