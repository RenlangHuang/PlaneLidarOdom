cmake_minimum_required(VERSION 3.9)
project(planelo VERSION 1.0)


set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)


find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  roscpp
  rospy
  rosbag
  image_transport
  cv_bridge
  tf
)

# g2o
list( APPEND CMAKE_MODULE_PATH /home/jacko/Downloads/g2o/cmake_modules)
find_package(G2O REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")
#find_package(Open3D REQUIRED)

# LibTorch
#set(Torch_DIR "/home/jacko/Downloads/libtorch-cxx11-abi-shared-with-deps-1.7.0+cu110/libtorch/share/cmake/Torch")
set(Torch_DIR "/home/jacko/Downloads/libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

#find_package(Eigen3 REQUIRED)
#find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
set(FMT_LIBRARIES fmt::fmt)

SET(G2O_LIBRARIES
    ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_CLI_LIBRARY}
    ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE} ${G2O_SOLVER_PCG} ${G2O_SOLVER_EIGEN}
    ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY}
    ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA} ${G2O_TYPES_SCLAM2D}
    ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D}
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    #${OpenCV_INCLUDE_DIRS}
    #${CERES_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)


add_executable(kittiPublisher src/kittiPublisher.cpp)
target_link_libraries(kittiPublisher ${catkin_LIBRARIES} ${PCL_LIBRARIES}) #${OpenCV_LIBS})

add_executable(mulranPublisher src/mulranPublisher.cpp)
target_link_libraries(mulranPublisher ${catkin_LIBRARIES} ${PCL_LIBRARIES}) #${OpenCV_LIBS})

#add_executable(Preprocessing src/Preprocessing.cpp)
#target_link_libraries(Preprocessing ${catkin_LIBRARIES} ${PCL_LIBRARIES}) #${TORCH_LIBRARIES}

#add_executable(demo src/ExtractPlaneDemo.cpp)
add_executable(demo src/main.cpp)
target_link_libraries(demo ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb) #${TORCH_LIBRARIES}

#catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
