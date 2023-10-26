cmake_minimum_required(VERSION 3.16)
project(my_work)

######################
### Cmake flags
######################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14 -O3 -w")
set (CMAKE_CXX_STANDARD 14)
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
find_package(SuiteSparse REQUIRED)

#####################
### include files

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ./src/VIO
  ./src/LIO
  ./src/common
  ./src/common/FOV_Checker
  ./src/common/kd_tree
  ./src/common/tools
  ./src/common/utility
  ./src/common/tools/r2live_sophus
)

catkin_package(DEPENDS EIGEN3 PCL SUITESPARSE
   )

######################
### src adding
######################
file(GLOB_RECURSE VISUAL_FEATURE_FILES
    "src/VIO/visual_feature/*.cpp"
    "src/VIO/visual_feature/camera_models/*.cc"
)

file(GLOB_RECURSE LIO_FILES
    "src/LIO/*.cpp"  #递归搜索src目录下的所有.cpp文件
    "src/common/*.cpp"

)
list(REMOVE_ITEM LIO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/LIO/feature_extract.cpp)
# Visual Feature Tracker
add_executable(feature_tracker ${VISUAL_FEATURE_FILES})
target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

######################
### lidar odometry
######################
add_executable(lio_feat_extract src/LIO/feature_extract.cpp)
target_link_libraries(lio_feat_extract ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(fastlio_mapping ${LIO_FILES})
target_link_libraries(fastlio_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${SUITESPARSE_LIBRARIES})
