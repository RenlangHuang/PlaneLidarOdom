#include <thread>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <pcl/filters/uniform_sampling.h>

#include "slam/AdamMap.h"



struct AdaptiveThreshold {
public:
    explicit AdaptiveThreshold(double initial_threshold, double min_motion_th, double max_range, double ema_weight=0.0)
        : initial_threshold_(initial_threshold), min_motion_th_(min_motion_th), max_range_(max_range), ema_weight_(ema_weight)
    {
        curr_threshold_ = initial_threshold;
        ema_update = (ema_weight > 1e-6);
        if (ema_update)
            model_error_sse2_ = initial_threshold * initial_threshold;
        model_deviation_ = Sophus::SE3d(Eigen::Matrix4d::Identity());
    }

    double GetThreshold() {
        return curr_threshold_;
    }

    // Update the current belief of the deviation from the prediction model
    inline void UpdateModelDeviation(const Sophus::SE3d &current_deviation) {
        model_deviation_ = current_deviation;
        curr_threshold_ = ComputeThreshold();
    }

    double ComputeModelError(const Sophus::SE3d &model_deviation, double max_range) {
        const double theta = Eigen::AngleAxisd(model_deviation.unit_quaternion()).angle();
        const double delta_rot = 2.0 * max_range * sin(theta / 2.0);
        const double delta_trans = model_deviation.translation().norm();
        return delta_trans + delta_rot;
    }

    double ComputeThreshold(){
        double model_error = ComputeModelError(model_deviation_, max_range_);
        if (model_error > min_motion_th_) {
            if (ema_update)
                model_error_sse2_ = (1 - ema_weight_) * model_error_sse2_ + ema_weight_ * model_error * model_error;
            else model_error_sse2_ += model_error * model_error;
            num_samples_++;
        }
        if (num_samples_ < 1)
            return initial_threshold_;
        else if (ema_update)
            return std::sqrt(model_error_sse2_);
        else return std::sqrt(model_error_sse2_ / num_samples_);
    }

private:
    // configurable parameters
    double initial_threshold_;
    double min_motion_th_;
    double max_range_;
    double ema_weight_;
    bool ema_update;

    // Local cache for computation
    double curr_threshold_;
    int num_samples_ = 0;
    double model_error_sse2_ = 0;
    Sophus::SE3d model_deviation_;
};


AdamMap map;
ros::Publisher publisher;

std::mutex mBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> pcdBuf;
std::queue<nav_msgs::Odometry::ConstPtr> poseBuf;

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserPoints)
{
	mBuf.lock();
	pcdBuf.push(laserPoints);
	mBuf.unlock();
}
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	poseBuf.push(laserOdometry);
	mBuf.unlock();
}

void publish()
{
	ros::Rate r(5);
	size_t frame_count = 0;
	const int inactive_publish_round = 3;

    visualization_msgs::Marker planes;
    planes.header.frame_id = "lidar_init";
    planes.ns = "ActiveMap";
    planes.id = 0;
    planes.type = visualization_msgs::Marker::TRIANGLE_LIST;
    planes.scale.x = 1.0;
    planes.scale.y = 1.0;
    planes.scale.z = 1.0;
    planes.pose.orientation.x = 0.0;
    planes.pose.orientation.y = 0.0;
    planes.pose.orientation.z = 0.0;
    planes.pose.orientation.w = 1.0;
    planes.action = visualization_msgs::Marker::ADD;
    planes.color.a = 0.1;//1.0;

	while (ros::ok()) {
		sensor_msgs::PointCloud2 laser_active_map_msg, laser_inactive_map_msg;
		pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_active_map(new pcl::PointCloud<pcl::PointXYZI>());
		pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_inactive_map(new pcl::PointCloud<pcl::PointXYZI>());
		std::vector<Frame *> vpKFs = map.GetKeyFrame();
		if (vpKFs.size() > frame_count)
		{
			frame_count = vpKFs.size();
            map.visualize(planes);
            planes.header.stamp = ros::Time::now();
            map.inactive_map_.header.stamp = planes.header.stamp;
            //publisher.publish(planes);
			if (frame_count % inactive_publish_round)
				map.visualize(publisher);
			else publisher.publish(planes);
		}
		r.sleep();
	}
}

void PlaneBundleAdjustment()
{
    ros::Rate r(10);
    size_t mnFrames = 0;
    int MAX_ITERATIONS_ = 1;//3;
    size_t MAX_QUEUE_SIZE_ = 3;
    double ESTIMATION_THRESHOLD_ = 0.0001;
    SummaryBoard board;
    while (ros::ok()) {
		if (map.GetNumOfKeyFrame() > mnFrames)
		{
            map.RemovePointsFarFromLocation();
            if ((int)map.GetNumOfKeyFrame() > map.mnMinSlidingWindowSize) {
                for (int i = 0; i < MAX_ITERATIONS_; i++)
                {
                    if (map.GetNumOfKeyFrame() > MAX_QUEUE_SIZE_ + mnFrames) break;
                    TicToc timer; double dx = map.PlaneBundleAdjustment();
                    printf("plane BA iteration %d: lie increment %f, time %.2f ms\n", i+1, dx, timer.toc());
                    board.update("plane BA per iteration", timer.toc());
                    if (dx < ESTIMATION_THRESHOLD_) break;
                }
            }
            mnFrames = map.GetNumOfKeyFrame();
		}
		r.sleep();
	}
    board.summarize();
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "main");
	ros::NodeHandle nh;

    bool use_ba = true;
	std::string save_dir;
	nh.getParam("save_directory", save_dir);
	nh.getParam("BA_sliding_window_size", map.mnSlidingWindowSize);
	nh.getParam("BA_minimum_sliding_window_size", map.mnMinSlidingWindowSize);
    nh.getParam("BA_Levenberg_Marquardt_coefficient", map.levenberg_marquardt);
    nh.getParam("BA", use_ba);

	ros::Subscriber s3 = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
	ros::Subscriber s5 = nh.subscribe<nav_msgs::Odometry>("/odometry_gt", 100, laserOdometryHandler);

	ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/point_cluster", 100);
	ros::Publisher pub2 = nh.advertise<sensor_msgs::PointCloud2>("/plane_cluster", 100);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path> ("/slam", 5);
    publisher = nh.advertise<visualization_msgs::Marker>("Map", 20);

    pcl::UniformSampling<pcl::PointXYZI> uniformFilter;
    
    AdaptiveThreshold mThreshold(1.0, 0.2, 100.0);

    visualization_msgs::Marker planes;
    planes.header.frame_id = "lidar_init";
    planes.ns = "PlaneFeatures";
    planes.id = 0;
    planes.type = visualization_msgs::Marker::TRIANGLE_LIST;
    planes.scale.x = 1.0;
    planes.scale.y = 1.0;
    planes.scale.z = 1.0;
    planes.pose.orientation.x = 0.0;
    planes.pose.orientation.y = 0.0;
    planes.pose.orientation.z = 0.0;
    planes.pose.orientation.w = 1.0;
    planes.action = visualization_msgs::Marker::ADD;
    planes.color.a = 0.1;//1.0;

	ros::Rate rate(30);
    std::thread publish_thread{publish};
    if (use_ba)
    {
        std::thread bundle_adjustment_thread{PlaneBundleAdjustment};
        bundle_adjustment_thread.detach();
    }
    SummaryBoard board;

	while(ros::ok())
	{
		while (!pcdBuf.empty() && !poseBuf.empty())
		{
			mBuf.lock();
			const double ts = poseBuf.front()->header.stamp.toSec();
			while (!pcdBuf.empty() && pcdBuf.front()->header.stamp.toSec() < ts) pcdBuf.pop();
			if (pcdBuf.empty()) { rate.sleep(); continue; }
			if (pcdBuf.front()->header.stamp.toSec() != ts) {
				printf("unsync message!\n");
				mBuf.unlock(); break;
			}
			pcl::PointCloud<pcl::PointXYZI>::Ptr ply(new pcl::PointCloud<pcl::PointXYZI>());
			pcl::fromROSMsg(*pcdBuf.front(), *ply);

			Eigen::Vector3d t_w_curr;
			Eigen::Quaterniond q_w_curr;

			q_w_curr.x() = poseBuf.front()->pose.pose.orientation.x;
			q_w_curr.y() = poseBuf.front()->pose.pose.orientation.y;
			q_w_curr.z() = poseBuf.front()->pose.pose.orientation.z;
			q_w_curr.w() = poseBuf.front()->pose.pose.orientation.w;
			t_w_curr.x() = poseBuf.front()->pose.pose.position.x;
			t_w_curr.y() = poseBuf.front()->pose.pose.position.y;
			t_w_curr.z() = poseBuf.front()->pose.pose.position.z;

			poseBuf.pop();
            pcdBuf.pop();
			mBuf.unlock();
            TicToc timer;

            // Tracking: point-to-plane frame-to-map registration
            pcl::PointCloud<pcl::PointXYZI> GridSubsampledCloud;
            uniformFilter.setRadiusSearch(0.5);
            uniformFilter.setInputCloud(ply);
            uniformFilter.filter(GridSubsampledCloud);

            std::vector<Eigen::Vector3d> curr_pcd;
            curr_pcd.reserve(GridSubsampledCloud.size());
            for (size_t i = 0; i < GridSubsampledCloud.size(); i++)
            {
                pcl::PointXYZI pt = GridSubsampledCloud.at(i);
                curr_pcd.push_back(Eigen::Vector3d(pt.x,pt.y,pt.z));
            }
            Sophus::SE3d pose_pred(Eigen::Matrix4d::Identity());
            std::vector<Frame *> vpKFs = map.GetKeyFrame();
            if (!vpKFs.empty())
            {
                pose_pred = vpKFs.back()->GetPose();
                if (vpKFs.size() > 1)
                {
                    pose_pred = pose_pred * vpKFs[vpKFs.size()-2]->GetPose().inverse() * pose_pred;
                }
                Sophus::SE3d pose(pose_pred);
                double th = mThreshold.ComputeThreshold();
                for (int i = 0; i < 200; i++)
                {
                    Sophus::SE3d delta_pose = map.PointToPlaneOptimization(curr_pcd, th, &pose);
                    pose = delta_pose * pose;
                    if (delta_pose.log().norm() < 1e-4) {
                        printf("optimize %d\n",i); break;
                    }
                }
                t_w_curr = pose.translation();
                q_w_curr = pose.unit_quaternion();
                mThreshold.UpdateModelDeviation(pose * pose_pred.inverse());
                printf("\033[1;32madaptive threshold %fm \033[0m\n", mThreshold.GetThreshold());
            }
            map.AddKeyFrame(new Frame(ts, t_w_curr, q_w_curr), ply);
            //map.RemovePointsFarFromLocation(t_w_curr);

            printf("[%f] track local map %f ms\n", ts, timer.toc());
            board.update("tracking", timer.toc());
            
            nav_msgs::Path laserPath;
			laserPath.header.stamp = ros::Time().fromSec(ts);
			laserPath.header.frame_id = "lidar_init";
			for (Frame *pKF : vpKFs)
			{
				Sophus::SE3d pose = pKF->GetPose();
				Eigen::Vector3d t_w_curr = pose.translation();
				Eigen::Quaterniond q_w_curr = pose.unit_quaternion();

				geometry_msgs::PoseStamped laserPose;
				laserPose.header.stamp = laserPath.header.stamp;
				laserPose.header.frame_id = "lidar_init";
				laserPose.pose.orientation.x = q_w_curr.x();
				laserPose.pose.orientation.y = q_w_curr.y();
				laserPose.pose.orientation.z = q_w_curr.z();
				laserPose.pose.orientation.w = q_w_curr.w();
				laserPose.pose.position.x = t_w_curr.x();
				laserPose.pose.position.y = t_w_curr.y();
				laserPose.pose.position.z = t_w_curr.z();
				laserPath.poses.push_back(laserPose);
			}
            pubPath.publish(laserPath);
        }
		rate.sleep();
		ros::spinOnce();
	}
    board.summarize();

    std::ofstream poses, stamps;
	poses.open(save_dir + ".txt", std::ios::out);
	stamps.open(save_dir + "_stamp.txt", std::ios::out);
	poses.setf(std::ios::scientific, std::ios::floatfield);
	stamps.setf(std::ios::scientific, std::ios::floatfield);
	poses.precision(8);
	stamps.precision(8);

    std::vector<Frame *> vpKFs = map.GetKeyFrame();
	for (auto vit=vpKFs.begin(), vend=vpKFs.end(); vit!=vend; vit++)
	{
		stamps << (*vit)->mTimeStamp << std::endl;
		Sophus::Matrix<double, 3, 4> T = (*vit)->GetPose().matrix3x4();
		poses << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
			  << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
			  << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << " " << std::endl;	
	}
	stamps.close();
	poses.close();
    
	return 0;
}
