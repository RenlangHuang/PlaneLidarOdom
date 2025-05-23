#include <cmath>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>


std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ios::in | std::ios::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_publisher");
    ros::NodeHandle n;
    double publish_delay = 0.1;
    double MINIMUM_RANGE = 2.5;
    double MAXIMUM_RANGE = 80.0;
    double ZMIN_RANGE = -5.0;
    std::string dataset_folder, sequence_number;

    int start_frame_id;
    
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence_number);
    n.getParam("maximum_range", MAXIMUM_RANGE);
    n.getParam("minimum_range", MINIMUM_RANGE);
    n.getParam("publish_delay", publish_delay);
    n.getParam("start_frame_id", start_frame_id);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    std::cout << "Publish interval: " << publish_delay << '\n';

    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 20);
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "lidar_init";
    odomGT.child_frame_id = "/ground_truth";
    
    nav_msgs::Path pathGT;
    pathGT.header.frame_id = "lidar_init";

    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string ground_truth_path = "results/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);


    Eigen::Matrix4d calib;
    std::string calib_path = "sequences/" + sequence_number + "/calib.txt";
    std::ifstream calib_file(dataset_folder + calib_path, std::ifstream::in);
    std::string calib_line;
    calib.block<1, 4>(3, 0) << 0, 0, 0, 1;
    while(std::getline(calib_file, calib_line)){
        if (calib_line[0]=='P') continue;
        else {
            std::stringstream sss(calib_line.substr(4));
            sss >> calib(0,0) >> calib(0,1) >> calib(0,2) >> calib(0,3)
                >> calib(1,0) >> calib(1,1) >> calib(1,2) >> calib(1,3)
                >> calib(2,0) >> calib(2,1) >> calib(2,2) >> calib(2,3);
            break;
        }
    }

    std::string line;
    std::size_t line_num = 0;
    ros::Rate rate(1.0 / publish_delay);
    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();

    while (std::getline(timestamp_file, line) && ros::ok()) {
        float timestamp = stof(line);
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 4, 4> gt_pose;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }
        gt_pose.block<1, 4>(3, 0) << 0, 0, 0, 1;
        gt_pose = calib.inverse() * gt_pose * calib;
        if (line_num < start_frame_id)
        {
            line_num++;
            continue;
        }
        else if (line_num == start_frame_id)
        {
            tf = gt_pose.inverse();
        }
        gt_pose = tf * gt_pose;
        Eigen::Quaterniond q(gt_pose.topLeftCorner<3, 3>());
        Eigen::Vector3d t = gt_pose.topRightCorner<3, 1>();
        q.normalize();

        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        pubOdomGT.publish(odomGT);

        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT);
        pubPathGT.publish(pathGT);

        // read lidar point cloud
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());

        pcl::PointCloud<pcl::PointXYZI>::Ptr laser_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4) {
            pcl::PointXYZI point;
            double range = sqrt(lidar_data[i]*lidar_data[i] + lidar_data[i + 1]*lidar_data[i + 1]);
            if (range > MINIMUM_RANGE && range < MAXIMUM_RANGE && lidar_data[i + 2] > ZMIN_RANGE) {
                Eigen::Vector3d _point(lidar_data[i], lidar_data[i + 1], lidar_data[i + 2]);
                const Eigen::Vector3d rotationVector = _point.cross(Eigen::Vector3d(0., 0., 1.));
                _point = Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * _point;
                point.x = _point[0];
                point.y = _point[1];
                point.z = _point[2];
                point.intensity = lidar_data[i + 3];
                laser_cloud->push_back(point);
            }
        }
        printf("[%f] totally %ld points in this lidar frame\n", timestamp, laser_cloud->size());

        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(*laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "lidar_init";
        pub_laser_cloud.publish(laser_cloud_msg);
        
        line_num ++;
        rate.sleep();
    }
    std::cout << "Done \n";
    return 0;
}
