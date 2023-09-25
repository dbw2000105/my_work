cmake_minimum_required(VERSION 3.16)
project(my_work)

######################
### Cmake flags
######################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++17 -O3 -w")
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

######################
### Packages
######################
set(cv_bridge_1_DIR /usr/local/share/cv_bridge_1/cmake)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge_1
    message_generation
    livox_ros_driver
    pcl_conversions
    )


find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
find_package(OpenCV 3.4.5 REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)

find_package(PythonLibs REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

######################
### Message generation
######################
add_message_files(
  FILES
  Pose6D.msg
  States.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ./src/VIO
  ./src/LIO
  ./src/common
)

catkin_package(
  DEPENDS PCL
)

######################
### src adding
######################
file(GLOB visual_feature_files
    "src/VIO/visual_feature/*.cpp"
    "src/VIO/visual_feature/camera_models/*.cc"
)

file(GLOB lio_files
    "src/LIO/*.cpp"
    "src/LIO/include/kd_tree/*.cpp"
)

# Visual Feature Tracker
add_executable(feature_tracker ${visual_feature_files})
target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

######################
### lidar odometry
######################
# add_executable(fastlio_mapping src/laserMapping.cpp ${lio_files})
# target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES})
# target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
