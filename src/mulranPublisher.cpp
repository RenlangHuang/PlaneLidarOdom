#define FMT_HEADER_ONLY
#include "fmt/format.h"

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

#include <sophus/se3.hpp>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>


Sophus::SE3d convert(Eigen::Matrix4d pose)
{
    Eigen::Quaterniond q(Eigen::Matrix3d(pose.topLeftCorner(3,3)));
    Eigen::Vector3d t(pose(0,3), pose(1,3), pose(2,3));
    return Sophus::SE3d(q.normalized(), t);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mulran_publisher");
    ros::NodeHandle n;
    int start_frame_id = 0;
    double publish_delay = 0.1;
    double MINIMUM_RANGE = 2.5;
    double MAXIMUM_RANGE = 120.0;
    std::string dataset_folder, sequence, save_dir;
    
    
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence);
	n.getParam("save_directory", save_dir);
    n.getParam("maximum_range", MAXIMUM_RANGE);
    n.getParam("minimum_range", MINIMUM_RANGE);
    n.getParam("publish_delay", publish_delay);
    n.getParam("start_frame_id", start_frame_id);
    std::cout << "Reading sequence " << sequence << " from " << dataset_folder << '\n';
    std::cout << "Publish interval: " << publish_delay << '\n';

    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 20);
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    
    nav_msgs::Path pathGT;
    pathGT.header.frame_id = "lidar_init";

    uint64_t timestamp;
    size_t line_num = 0;
    const float sweep_duration = 0.1;
    const float at = -0.5 * sweep_duration;
    const std::string dir = dataset_folder + sequence + "/sensor_data/Ouster/";
    FILE *file = fopen((dataset_folder + sequence + "/sensor_data/ouster_front_stamp.csv").c_str(), "r");

    ros::Rate rate(1.0 / publish_delay);
    Eigen::Matrix4d gt_pose = Eigen::Matrix4d::Identity();

    file = fopen((dataset_folder + sequence + "/global_pose.csv").c_str(), "r");
    std::vector<uint64_t> vTimestamp;
    std::vector<Sophus::SE3d> vPoses;

    auto readGT = [&](FILE *f){
        return fscanf(f, "%ld,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", &timestamp,
                      &gt_pose(0,0), &gt_pose(0,1), &gt_pose(0,2), &gt_pose(0,3),
                      &gt_pose(1,0), &gt_pose(1,1), &gt_pose(1,2), &gt_pose(1,3),
                      &gt_pose(2,0), &gt_pose(2,1), &gt_pose(2,2), &gt_pose(2,3));
    };
    
    Eigen::Quaterniond r(Eigen::AngleAxisd(0.0001 * M_PI / 180.0, Eigen::Vector3d(1,0,0)));
    Eigen::Quaterniond p(Eigen::AngleAxisd(0.0003 * M_PI / 180.0, Eigen::Vector3d(0,1,0)));
    Eigen::Quaterniond y(Eigen::AngleAxisd(179.6654*M_PI / 180.0, Eigen::Vector3d(0,0,1)));
    
    Sophus::SE3d calibLidar2Base(y * p * r, Eigen::Vector3d(1.7042, -0.021, 1.8047));
    Sophus::SE3d calibBase2Lidar = calibLidar2Base.inverse();
    Sophus::SE3d global_init;

    while (readGT(file) == 13 && ros::ok())
    {
        Eigen::Quaterniond q(1, 0, 0, 0);
        Eigen::Vector3d t(0, 0, 0);
        if (line_num > 0)
        {
            Sophus::SE3d global_curr = global_init.inverse() * convert(gt_pose);
            Sophus::SE3d local_curr_ouster = calibLidar2Base * global_curr * calibBase2Lidar;
            q = local_curr_ouster.unit_quaternion();
            t = local_curr_ouster.translation();
        }
        else global_init = convert(gt_pose);
        vPoses.push_back(Sophus::SE3d(q, t));
        vTimestamp.push_back(timestamp);
        line_num++;
    }
    fclose(file);

    file = fopen((dataset_folder + sequence + "/sensor_data/ouster_front_stamp.csv").c_str(), "r");
    size_t index = 0;
    line_num = 0;

    while (fscanf(file,"%ld\n", &timestamp) == 1 && ros::ok())
    {
        while (index<vPoses.size() && vTimestamp[index]<timestamp) index++;
        if (index >= vPoses.size()) break;
        if (index == 0) continue;
        
        double t_l = (double)vTimestamp[index-1] / 1e9;
        double t_n = (double)vTimestamp[index] / 1e9;
        double dt = ((double)timestamp / 1e9 - t_l) / (t_n - t_l);
        Eigen::Quaterniond q = vPoses[index-1].unit_quaternion().slerp(dt, vPoses[index].unit_quaternion());
        Eigen::Vector3d t = vPoses[index-1].translation() * (1 - dt) + vPoses[index].translation() * dt;
        if (pathGT.poses.size() == 0)
        {
            global_init = Sophus::SE3d(q, t);
            q = Eigen::Quaterniond(1, 0, 0, 0);
            t = Eigen::Vector3d(0, 0, 0);
        }
        else
        {
            Sophus::SE3d global_curr = global_init.inverse() * Sophus::SE3d(q, t);
            q = global_curr.unit_quaternion();
            t = global_curr.translation();
        }
        geometry_msgs::PoseStamped odomGT;
        odomGT.header.frame_id = "lidar_init";
        odomGT.header.stamp = ros::Time().fromNSec(timestamp);
        odomGT.pose.orientation.x = q.x();
        odomGT.pose.orientation.y = q.y();
        odomGT.pose.orientation.z = q.z();
        odomGT.pose.orientation.w = q.w();
        odomGT.pose.position.x = t(0);
        odomGT.pose.position.y = t(1);
        odomGT.pose.position.z = t(2);

        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(odomGT);

        std::ifstream lidar_data_file(dir + std::to_string(timestamp) + ".bin", std::ios::in | std::ios::binary);
        if (!lidar_data_file.is_open())
        {
            std::cout << dir + std::to_string(timestamp) + ".bin" << " NOT EXISTS!\n";
            continue;
        }
        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
        lidar_data_file.seekg(0, std::ios::beg);
        std::vector<float> lidar_data(num_elements);
        lidar_data_file.read(reinterpret_cast<char*>(&lidar_data[0]), num_elements*sizeof(float));
        lidar_data_file.close();

        pcl::PointCloud<pcl::PointXYZI>::Ptr laser_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (std::size_t i = 0; i < lidar_data.size(); i += 4) {
            pcl::PointXYZI point;
            double range = sqrt(lidar_data[i] * lidar_data[i] + lidar_data[i+1] * lidar_data[i+1]);
            if (range > MINIMUM_RANGE)
            {
                size_t cnt = i / 4;
                size_t col = cnt / 64;
                point.x = lidar_data[i];
                point.y = lidar_data[i + 1];
                point.z = lidar_data[i + 2];
                point.intensity = at + sweep_duration * (float)col / 1024.0;
                // point.intensity = lidar_data[i+3];
                // point.ring = cnt % 64; // ring
                laser_cloud->push_back(point);
            }
        }
        printf("[%lf] totally %ld points in this lidar frame\n", (double)timestamp/1e9, laser_cloud->size());

        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(*laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromNSec(timestamp);
        laser_cloud_msg.header.frame_id = "lidar_init";
        pub_laser_cloud.publish(laser_cloud_msg);
        
        pubPathGT.publish(pathGT);
        line_num ++;
        rate.sleep();
    }
    fclose(file);
    std::cout << "Done \n";

    std::ofstream poses, stamps;
	poses.open(save_dir + "_gt.txt", std::ios::out);
	stamps.open(save_dir + "_stamp_gt.txt", std::ios::out);
	poses.setf(std::ios::scientific, std::ios::floatfield);
	stamps.setf(std::ios::scientific, std::ios::floatfield);
	poses.precision(8);
	stamps.precision(8);

	for (geometry_msgs::PoseStamped pose_stamp : pathGT.poses)
	{
		stamps << pose_stamp.header.stamp.toSec() << std::endl;
        Eigen::Vector3d t(pose_stamp.pose.position.x,
                          pose_stamp.pose.position.y,
                          pose_stamp.pose.position.z);
        Eigen::Quaterniond q(pose_stamp.pose.orientation.w,
                             pose_stamp.pose.orientation.x,
                             pose_stamp.pose.orientation.y,
                             pose_stamp.pose.orientation.z);
		Sophus::Matrix<double, 3, 4> T = Sophus::SE3d(q,t).matrix3x4();
		poses << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
			  << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
			  << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << " " << std::endl;	
	}
	stamps.close();
	poses.close();
    return 0;
}
