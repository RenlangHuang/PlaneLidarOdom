## Real-time Plane Lidar Odometry with Efficient Bundle Adjustment
This repository presents a simple yet effective real-time lidar odometry and mapping system built upon plane features and bundle adjustment proposed by [BALM2](https://github.com/hku-mars/BALM) but much easier to follow.

<div align="center">
  <img src=figures/PlaneLO.png width=60%/>
</div>

## 1. Installation
### 1.1. **Ubuntu** and **ROS**
Ubuntu 64-bit 16.04 or 18.04 or 20.04 (tested).

ROS Kinetic or Melodic or Noetic (tested). [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL**
* Run these commands to install dependencies first:
```
sudo apt-get update
sudo apt-get install git build-essential linux-libc-dev -y
sudo apt-get install cmake -y
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev -y
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common -y
sudo apt-get install libflann1.9 libflann-dev -y
sudo apt-get install libeigen3-dev -y
sudo apt-get install libboost-all-dev -y
sudo apt-get install libvtk7.1p-qt libvtk7.1p libvtk7-qt-dev -y
sudo apt-get install libqhull* libgtest-dev -y
sudo apt-get install freeglut3-dev pkg-config -y
sudo apt-get install libxmu-dev libxi-dev -y
sudo apt-get install mono-complete -y
sudo apt-get install openjdk-8-jdk openjdk-8-jre -y
```           
* Then follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html). PCL 1.12.0 has been tested.

### 1.3. **FMT** and **Sophus**

Install FMT, the prerequisite of Sophus:
```
git clone https://github.com/fmtlib/fmt.git
cd fmt
mkdir build
cd build
cmake ..
make
sudo make install
```

Clone the repository and make:

```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build
cd build
cmake ..
make
sudo make install
```

### 1.4. **TBB**

Follow [oneTBB installation](https://github.com/oneapi-src/oneTBB/blob/master/INSTALL.md). Release tagged 2021.8.0 is tested.

## 2. Build PlaneLO
Clone the repository and catkin_make:

```
mkdir -p ~/planelo_ws/src
cd ~/planelo_ws/src
git clone https://github.com/NeSC-IV/KDD-LOAM.git
cd ../
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## 3. Datasets Preparation Example (KITTI)

Download the data from the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to `YOUR_DATA_PATH`, which you are supposed to modify in `/TCKDD/datasets/kitti.py` and `/launch/kitti_publisher.launch`. The data should be organized as follows:

```text
--YOUR_DATA_PATH--KITTI_data_odometry--results (ps: ground truth)
                                    |--sequences--00--calib.txt
                                    |          |  |--times.txt
                                    |          |--...
                                    |--velodyne--sequences--00--velodyne--000000.bin
                                              |              |         |--...
                                              |              |--...
```

## 4. Odometry and Mapping: KITTI Example (Velodyne HDL-64)
Download [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to `YOUR_DATA_PATH` and set the `dataset_folder` and `sequence_number` parameters in `kitti_publisher.launch` file. You can start the LiDAR odometry and mapping by running the following commands in different terminals to launch the corresponding groups of ROS nodes.
```
python keypointsDescription.py
python odometry.py
roslaunch kddloam_velodyne kddloam.launch
roslaunch kddloam_velodyne kitti_publisher.launch
```

Meanwhile, you can launch the `savePath` roscpp node (refer to `kitti_publisher.launch`) to record the localization results to a txt file, then you can evaluate the relative pose errors through the official [KITTI odometry evaluation tools](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool) after synchronizing the localization results and the ground-truth poses. Note that the performance reported in our paper is evaluated on the premise that no LiDAR frames are discarded during system operation.

### Acknowledgments
In this project we use (parts of) the official implementations of the following works: 

- [KISS-ICP](https://github.com/PRBonn/kiss-icp) (motion compensation, adaptive threshold, voxel map, robust registration)
- [BALM2](https://github.com/hku-mars/BALM) (point cluster, and plane BA)
- [tsl robin-map library](https://github.com/Tessil/robin-map) (from which the `include/tsl` is forked directly)

We thank the respective authors for open sourcing their methods. We would also like to thank reviewers.
