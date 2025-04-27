#ifndef ADAMMAP_H
#define ADAMMAP_H

#include <list>
#include <mutex>
#include <queue>
#include <vector>
#include <unordered_map>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/thread/shared_mutex.hpp>
#include <boost/thread.hpp>
#include <tsl/robin_map.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for_each.h>

#include "slam/common.h"


typedef Eigen::Vector3i Voxel;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
    }
};


class PointCluster {
public:
    Eigen::Matrix3d P;
    Eigen::Vector3d v;
    int N;

    PointCluster()
    {
        P.setZero();
        v.setZero();
        N = 0;
    }
    PointCluster(Eigen::Vector3d point)
    {
        P = point * point.transpose();
        v = point;
        N = 1;
    }
    Eigen::Matrix3d cov() const
    {
        Eigen::Vector3d center = v / (double)N;
        return P / (double)N - center*center.transpose();
    }
    void insert(const Eigen::Vector3d &point)
    {
        N++;
        v += point;
        P += point * point.transpose();
    }
    PointCluster &operator+=(const PointCluster &other)
    {
        this->P += other.P;
        this->v += other.v;
        this->N += other.N;
        return *this;
    }
    PointCluster operator*(Sophus::SE3d pose)
    {
        PointCluster transformed;
        Eigen::Matrix3d R = pose.rotationMatrix();
        Eigen::Vector3d t = pose.translation();
        Eigen::Matrix3d rvt = R * v * t.transpose();
        transformed.P = R * P * R.transpose() + rvt + rvt.transpose() + (double)N * t * t.transpose();
        transformed.v = R * v + (double)N * t;
        transformed.N = N;
        return transformed;
    }
    Eigen::Vector3d centroid() const
    {
        return v / (double)N;
    }
    std::pair<Eigen::Vector3d, Eigen::Vector3d> estimate() const
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov());
        Eigen::Vector3d normal = solver.eigenvectors().col(0);
        Eigen::Vector3d eigval = solver.eigenvalues();
        return std::make_pair(normal, eigval);
    }
    Eigen::Matrix4d coordinate() const
    {
        Eigen::Matrix4d coordinate_;
        coordinate_.block<3,3>(0,0) = P;
        coordinate_.block<3,1>(0,3) = v;
        coordinate_.block<1,3>(3,0) = v.transpose();
        coordinate_(3,3) = (double)N;
        return coordinate_;
    }
};


class Frame {
public:
    Frame (double timestamp, Eigen::Vector3d t, Eigen::Quaterniond q)
    {
        mnId = nNextId++;
        mPose = Sophus::SE3d(q.normalized(), t);
        mTimeStamp = timestamp;
    }
    std::vector<PointCluster> DownSample(pcl::PointCloud<pcl::PointXYZI>::Ptr points, double voxel_size)
    {
        tsl::robin_map<Voxel, PointCluster, VoxelHash> map_;
        std::for_each(points->begin(), points->end(), [&](pcl::PointXYZI point) {
            Eigen::Vector3d _point(point.x, point.y, point.z);
            _point = mPose * _point;
            auto voxel = Voxel((_point / voxel_size).template cast<int>());
            auto search = map_.find(voxel);
            if (search != map_.end()) search.value().insert(_point);
            else map_.insert({voxel, PointCluster{_point}});
        });
        std::vector<PointCluster> sampled;
        for (const auto &[voxel, voxel_block] : map_) {
            if (voxel_block.N >=3) sampled.push_back(voxel_block);
        }
        return sampled;
    }
    void SetPose(Eigen::Quaterniond q_w_curr, Eigen::Vector3d t_w_curr);
    void SetPose(Sophus::SE3d T_w_curr);
    Sophus::SE3d GetPose();

public:
    double mTimeStamp;
    static unsigned long nNextId;
    unsigned long mnId;

protected:
    Sophus::SE3d mPose;
    boost::mutex mMutexPose;
};
unsigned long Frame::nNextId = 1;
void Frame::SetPose(Eigen::Quaterniond q_w_curr, Eigen::Vector3d t_w_curr)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    mPose = Sophus::SE3d(q_w_curr.normalized(), t_w_curr);
}
void Frame::SetPose(Sophus::SE3d T_w_curr)
{
    boost::mutex::scoped_lock lock(mMutexPose);
    mPose = T_w_curr;
}
Sophus::SE3d Frame::GetPose()
{
    boost::mutex::scoped_lock lock(mMutexPose);
    return mPose;
}


class VoxelBlock {
public:
    VoxelBlock() : point_cluster_(), mMutex(new boost::shared_mutex) {}
    bool insert(Frame *pKF, PointCluster pc)
    {
        boost::unique_lock<boost::shared_mutex> lock(*mMutex);
        point_cluster_ += pc;
        mPointClusters[pKF] = pc * pKF->GetPose().inverse();
        Eigen::Vector3d eigval = point_cluster_.estimate().second;
        return eigval[0] < 0.1 * eigval[1] && eigval[0] < 1.0;
    }
    Eigen::Vector3d centroid()
    {
        boost::shared_lock<boost::shared_mutex> lock(*mMutex);
        return point_cluster_.centroid();
    }
    std::pair<Eigen::Vector3d, Eigen::Vector3d> estimate()
    {
        boost::shared_lock<boost::shared_mutex> lock(*mMutex);
        return point_cluster_.estimate();
    }
    double N() const { return (double)point_cluster_.N; }
    void SetBadFlag()
    {
        if (mbBad) return;
        boost::unique_lock<boost::shared_mutex> lock(*mMutex);
        mPointClusters.clear();
        mbBad = true;
    }
    JacobianHessianTuple CalJacobianHessian(const std::unordered_map<Frame*, int> &mpKFs) const;
    void OptimizeLandmark();

public:
    bool mbBad = false;
    unsigned long mnBALocalForKF = 0;

protected:
    PointCluster point_cluster_;
    std::map<Frame *, PointCluster> mPointClusters;
    std::shared_ptr<boost::shared_mutex> mMutex;
};

JacobianHessianTuple VoxelBlock::CalJacobianHessian(const std::unordered_map<Frame*, int> &mpKFs) const
{
    JacobianHessianTuple JH((int)mpKFs.size());
    if (mPointClusters.size() <= 1) return JH;
    auto &[Hessian, Jacobian] = JH;

    Eigen::Matrix4d C;
    std::vector<std::pair<Frame*,PointCluster> > point_clusters;
    {
        boost::shared_lock<boost::shared_mutex> lock(*mMutex);
        C = point_cluster_.coordinate();
        for (auto [pKF, point_cluster] : mPointClusters)
        {
            if (mpKFs.count(pKF))
                point_clusters.push_back({pKF, point_cluster});
        }
    }
    double N = C(3,3); C = C / N;
    Eigen::Vector3d v_bar = C.block<3,1>(0,3);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3,3>(0,0) - v_bar*v_bar.transpose());
    Eigen::Vector3d ld = saes.eigenvalues();
    Eigen::Matrix3d u = saes.eigenvectors();
    Eigen::Matrix3d ulx = Sophus::SO3d::hat(u.col(0));
    std::vector<Eigen::Matrix<double,6,4> > V(3);
    std::vector<Eigen::Vector6d> VTCF((int)mpKFs.size());
    std::vector<Eigen::Vector6d> g_kl[3];
 
    if (ld[0] > ld[1] * 0.1 || ld[0] > 0.1) return JH;
    ld[1] = 2.0 / (ld[0] - ld[1]);
    ld[2] = 2.0 / (ld[0] - ld[2]);
    for (int k = 0; k < 3; k++)
    {
        V[k].setZero();
        V[k].block<3,3>(0,0) = Sophus::SO3d::hat(-u.col(k));
        V[k].block<3,1>(3,3) = u.col(k);
        g_kl[k].resize(mpKFs.size());
    }
    for (auto [pKFj, Cj] : point_clusters)
    {
        int j = mpKFs.find(pKFj)->second;
        Eigen::Matrix4d Tj = pKFj->GetPose().matrix();
        Eigen::Matrix4d TjCj = Tj * Cj.coordinate();
        Eigen::Matrix<double,6,4> VTCj = V[0] * TjCj;
        Eigen::Matrix<double,4,3> TC_TCFSpj = Tj.block<3,4>(0,0).transpose();
        TC_TCFSpj.block<1, 3>(3, 0) -= v_bar.transpose();
        TC_TCFSpj = TjCj * TC_TCFSpj;
        VTCF[j] = VTCj.block<6,1>(0,3);
        for (int k = 0; k < 3; k++)
        {
            g_kl[k][j] = (V[k] * TC_TCFSpj * u.col(0) + V[0] * TC_TCFSpj * u.col(k)) / N;
        }
        Eigen::Matrix3d SpTC_TCFSpUlx = TC_TCFSpj.block<3,3>(0,0);
        SpTC_TCFSpUlx.noalias() = Sophus::SO3d::hat(SpTC_TCFSpUlx * u.col(0) / N) * ulx;
        Hessian.block<3,3>(6*j,6*j) += SpTC_TCFSpUlx + SpTC_TCFSpUlx.transpose();
        Hessian.block<6,6>(6*j,6*j).noalias() += VTCj * (Tj.transpose() * (2.0/N)) * V[0].transpose();
        Jacobian.block<6,1>(6*j,0) += g_kl[0][j];
    }
    for (auto vit=point_clusters.begin(), vend=point_clusters.end(); vit!=vend; vit++)
    {
        int i = mpKFs.find(vit->first)->second;
        for (auto vit2=point_clusters.begin(); vit2!=point_clusters.end(); vit2++)
        {
            int j = mpKFs.find(vit2->first)->second; if (i > j) continue;
            Eigen::Matrix6d VTCFCTV_gTg = VTCF[i] * (-2.0/N/N) * VTCF[j].transpose();
            VTCFCTV_gTg.noalias() += g_kl[1][i] * ld[1] * g_kl[1][j].transpose();
            VTCFCTV_gTg.noalias() += g_kl[2][i] * ld[2] * g_kl[2][j].transpose();
            if (i != j) Hessian.block<6,6>(6*j,6*i) += VTCFCTV_gTg.transpose();
            Hessian.block<6,6>(6*i,6*j) += VTCFCTV_gTg;
        }
    }
    //Hessian *= N;
    //Jacobian *= N;
    return JH;
}

void VoxelBlock::OptimizeLandmark()
{
    PointCluster new_point_cluster;
    {
        boost::shared_lock<boost::shared_mutex> lock(*mMutex);
        for (auto [pKF, point_cluster] : mPointClusters)
            new_point_cluster += point_cluster * pKF->GetPose();
    }
    boost::unique_lock<boost::shared_mutex> lock(*mMutex);
    point_cluster_ = new_point_cluster;
}


class AdamMap
{
public:
    double voxel_size_ = 1.0;
    double max_distance_ = 100.0;
    double levenberg_marquardt = 0.0;
    int mnSlidingWindowSize = 3;
    int mnMinSlidingWindowSize = 3;//10;

    std::vector<Frame *> mvpKeyFrames;
    std::list<Frame *> mlpSlidingWindow;
    std::map<Frame *, std::set<std::shared_ptr<VoxelBlock> > > mspObservedPlanes;

    visualization_msgs::Marker inactive_map_, active_map_;
    tsl::robin_map<Voxel, std::shared_ptr<VoxelBlock>, VoxelHash> map_;

protected:
    boost::mutex mMutexFrame;
    boost::shared_mutex mMutexMap;
    boost::mutex mMutexInactiveMap;

public:
    AdamMap();
    
    Sophus::SE3d PointToPlaneOptimization(const std::vector<Eigen::Vector3d> &points, double max_distance, Sophus::SE3d *pose);
    double PlaneBundleAdjustment();
    
    std::vector<Frame *> GetKeyFrame() {boost::mutex::scoped_lock lock(mMutexFrame); return mvpKeyFrames;}
    size_t GetNumOfKeyFrame() const { return mvpKeyFrames.size(); }

    void AddKeyFrame(Frame *pKF, pcl::PointCloud<pcl::PointXYZI>::Ptr points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    void RemovePointsFarFromLocation();
    void visualize(visualization_msgs::Marker &planes);
    void visualize(ros::Publisher pub);
    void clear();
};


AdamMap::AdamMap()
{
    inactive_map_.header.frame_id = "lidar_init";
    inactive_map_.ns = "InactiveMap";
    inactive_map_.id = 0;
    inactive_map_.type = visualization_msgs::Marker::TRIANGLE_LIST;
    inactive_map_.scale.x = 1.0;
    inactive_map_.scale.y = 1.0;
    inactive_map_.scale.z = 1.0;
    inactive_map_.pose.orientation.x = 0.0;
    inactive_map_.pose.orientation.y = 0.0;
    inactive_map_.pose.orientation.z = 0.0;
    inactive_map_.pose.orientation.w = 1.0;
    inactive_map_.action=visualization_msgs::Marker::ADD;
    inactive_map_.color.a = 0.1;//1.0;
    //inactive_map_.color.g = 0.8;

    active_map_.header.frame_id = "lidar_init";
    active_map_.ns = "ActiveMap";
    active_map_.id = 0;
    active_map_.type = visualization_msgs::Marker::TRIANGLE_LIST;
    active_map_.scale.x = 1.0;
    active_map_.scale.y = 1.0;
    active_map_.scale.z = 1.0;
    active_map_.pose.orientation.x = 0.0;
    active_map_.pose.orientation.y = 0.0;
    active_map_.pose.orientation.z = 0.0;
    active_map_.pose.orientation.w = 1.0;
    active_map_.action = visualization_msgs::Marker::ADD;
    active_map_.color.a = 0.1;//1.0;
}
void AdamMap::clear() {
    boost::unique_lock<boost::shared_mutex> lock(mMutexMap);
    map_.clear();
}


Sophus::SE3d AdamMap::PointToPlaneOptimization(const std::vector<Eigen::Vector3d> &points, double max_distance, Sophus::SE3d *pose)
{
    auto search = [&](Eigen::Vector3d point, double radius)
    {
        auto closest_neighbor = map_.end();
        double closest_distance2 = radius * radius + voxel_size_ * voxel_size_ * 0.25;
        int kx = static_cast<int>(point[0] / voxel_size_);
        int ky = static_cast<int>(point[1] / voxel_size_);
        int kz = static_cast<int>(point[2] / voxel_size_);
        int search_radius_ = 1;

        for (int i = kx - search_radius_; i <= kx + search_radius_; ++i) {
            for (int j = ky - search_radius_; j <= ky + search_radius_; ++j) {
                for (int k = kz - search_radius_; k <= kz + search_radius_; ++k) {
                    auto search = map_.find(Voxel(i, j, k));
                    if (search == map_.end()) continue;
                    auto &voxel_block = search.value();
                    if (voxel_block->mbBad) continue;
                    double dist2 = (point - voxel_block->centroid()).squaredNorm();
                    if (dist2 < closest_distance2)
                    {
                        closest_distance2 = dist2;
                        closest_neighbor = search;
                    }
                }
            }
        }
        return closest_neighbor;
    };
    boost::shared_lock<boost::shared_mutex> read_lock(mMutexMap);
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [JTJ, JTe] = tbb::parallel_reduce(
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()}, JacobianTuple(),
        // 1st lambda: Parallel computation
        [&](const tbb::blocked_range<points_iterator> &r, JacobianTuple J) -> JacobianTuple {
            auto &[JTJ_private, JTe_private] = J;
            for (const auto &point : r) {
                Eigen::Vector3d _point = point;
                if (pose) _point = *pose * _point;
                auto closest_neighbours = search(_point, max_distance);
                if (closest_neighbours != map_.end()) {
                    Eigen::Vector3d center = closest_neighbours.value()->centroid();
                    Eigen::Vector3d normal = closest_neighbours.value()->estimate().first;
                    const double error = (_point - center).dot(normal);
                    if (abs(error) > max_distance) continue;

                    Eigen::Vector6d J_r;
                    J_r.block<3, 1>(0, 0) = normal;
                    J_r.block<3, 1>(3, 0) = Sophus::SO3d::hat(_point) * normal;
                    double w = max_distance / 9.0 + error * error; w = 1.0 / w / w;
                    JTJ_private.noalias() += w * J_r * J_r.transpose();
                    JTe_private.noalias() += w * J_r * error;
                }
            }
            return J;
        },
        // 2nd lambda: Parallel reduction
        [](JacobianTuple a, const JacobianTuple&b) -> JacobianTuple { return a + b; }
    );
    return Sophus::SE3d::exp(JTJ.ldlt().solve(-JTe));
}

double AdamMap::PlaneBundleAdjustment()
{
    std::vector<Frame *> vpKFs;
    std::unordered_map<Frame*, int> mpKFs;
    std::vector<std::shared_ptr<VoxelBlock> > voxel_blocks;
    voxel_blocks.reserve(map_.size());
    {
        boost::mutex::scoped_lock lock(mMutexFrame);
        if ((int)mlpSlidingWindow.size() < mnMinSlidingWindowSize) return 0;
        vpKFs.insert(vpKFs.end(), mlpSlidingWindow.begin(), mlpSlidingWindow.end());
        unsigned long mnBALocalForCurrentKF = vpKFs.back()->mnId;
        for (int i = 0; i < (int)vpKFs.size(); i++)
        {
            mpKFs[vpKFs[i]] = i;
            if (mspObservedPlanes.count(vpKFs[i]))
            {
                std::set<std::shared_ptr<VoxelBlock> > &spObservedPlanes = mspObservedPlanes[vpKFs[i]];
                for (std::shared_ptr<VoxelBlock> pvb : spObservedPlanes)
                {
                    if (pvb->mbBad || pvb->mnBALocalForKF == mnBALocalForCurrentKF) continue;
                    pvb->mnBALocalForKF = mnBALocalForCurrentKF;
                    voxel_blocks.push_back(pvb);
                }
            }
            else std::cout << "WARNING: KeyFrame without observed features\n";
        }
    }
    JacobianHessianTuple JH(vpKFs.size());
    JH = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, voxel_blocks.size()), JacobianHessianTuple((int)vpKFs.size()),
        [&](const tbb::blocked_range<size_t> &r, JacobianHessianTuple JH_) -> JacobianHessianTuple {
            for (size_t i=r.begin(); i!=r.end(); ++i)
            {
                JacobianHessianTuple JH_private = voxel_blocks[i]->CalJacobianHessian(mpKFs);
                //JH_private.DTHD /= N; JH_private.DTJT /= N;
                JH_ = JH_ + JH_private;
            }
            return JH_;
        }, [](JacobianHessianTuple a, const JacobianHessianTuple &b) -> JacobianHessianTuple { return a + b; }
    );
    tbb::parallel_for(size_t(0), vpKFs.size(), [&](size_t i){
        Eigen::Vector3d temp3x1 = JH.DTJT.block<3,1>(i*6,0);
        JH.DTJT.block<3,1>(i*6,0) = JH.DTJT.block<3,1>(i*6+3,0);
        JH.DTJT.block<3,1>(i*6+3,0) = temp3x1;
        for (size_t j = 0; j < vpKFs.size(); j++) {
            Eigen::Matrix3d temp3x3 = JH.DTHD.block<3,3>(i*6,j*6);
            JH.DTHD.block<3,3>(i*6,j*6) = JH.DTHD.block<3,3>(i*6+3,j*6+3);
            JH.DTHD.block<3,3>(i*6+3,j*6+3) = temp3x3;
            
            temp3x3 = JH.DTHD.block<3,3>(i*6+3,j*6);
            JH.DTHD.block<3,3>(i*6+3,j*6) = JH.DTHD.block<3,3>(i*6,j*6+3);
            JH.DTHD.block<3,3>(i*6,j*6+3) = temp3x3;
        }
    });
    //std::cout << JH.DTHD << std::endl << JH.DTJT.transpose() << std::endl;
    Eigen::MatrixXd D(6 * vpKFs.size(), 6 * vpKFs.size()); D.setZero();
    D.diagonal() = JH.DTHD.diagonal() * levenberg_marquardt;
    Eigen::VectorXd dx = (JH.DTHD + D).ldlt().solve(-JH.DTJT);
    for (size_t i = 0; i < vpKFs.size(); i++) {
        Sophus::SE3d pose = vpKFs[i]->GetPose();
        Eigen::Vector6d dxi = dx.block<6,1>(i*6, 0);
        vpKFs[i]->SetPose(Sophus::SE3d::exp(dxi) * pose);
    }
    boost::unique_lock<boost::shared_mutex> lock(mMutexMap);
    tbb::parallel_for(size_t(0), voxel_blocks.size(), [&voxel_blocks](size_t i){
        if (!voxel_blocks[i]->mbBad) voxel_blocks[i]->OptimizeLandmark();
    });
    return dx.norm();
}


void AdamMap::AddKeyFrame(Frame *pKF, pcl::PointCloud<pcl::PointXYZI>::Ptr points)
{
    std::set<std::shared_ptr<VoxelBlock> > spObservedPlanes;
    std::vector<PointCluster> planes = pKF->DownSample(points, voxel_size_);
    {
        boost::unique_lock<boost::shared_mutex> write_lock(mMutexMap);
        std::for_each(planes.cbegin(), planes.cend(), [&](PointCluster plane) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(plane.cov());
            Eigen::Vector3d eigval = solver.eigenvalues();
            if (eigval[0] > 0.1 * eigval[1] || eigval[0] > 1) return;

            auto voxel = Voxel((plane.centroid() / voxel_size_).template cast<int>());
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                auto &voxel_block = search.value();
                if (!voxel_block->mbBad) {
                    if (voxel_block->insert(pKF, plane))
                        spObservedPlanes.insert(voxel_block);
                    else voxel_block->SetBadFlag();
                }
            } else {
                auto pvb = std::make_shared<VoxelBlock>();
                pvb->insert(pKF, plane);
                map_.emplace(voxel, pvb);
                spObservedPlanes.insert(pvb);
            }
        });
    }
    {
        boost::mutex::scoped_lock lock(mMutexFrame);
        if (!mvpKeyFrames.empty())
        {
            mlpSlidingWindow.push_back(pKF);
            while ((int)mlpSlidingWindow.size() > mnSlidingWindowSize)
            {
                auto mit = mspObservedPlanes.find(mlpSlidingWindow.front());
                if (mit != mspObservedPlanes.end()) mit->second.clear();
                mspObservedPlanes.erase(mit);
                mlpSlidingWindow.pop_front();
            }
        }
        mvpKeyFrames.push_back(pKF);
        mspObservedPlanes[pKF] = spObservedPlanes;
    }
}

void AdamMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin)
{
    Eigen::Vector3d p1(0.5*voxel_size_,  0.5*voxel_size_, 0.0);
    Eigen::Vector3d p2(-0.5*voxel_size_, 0.5*voxel_size_, 0.0);
    Eigen::Vector3d p3(-0.5*voxel_size_, -0.5*voxel_size_, 0.0);
    Eigen::Vector3d p4(0.5*voxel_size_,  -0.5*voxel_size_, 0.0);
    Eigen::Vector3d z0(0.0, 0.0, 1.0);
    std::list<Voxel> removed;
    {
        boost::shared_lock<boost::shared_mutex> read_lock(mMutexMap);
        const auto max_distance2 = max_distance_ * max_distance_;
        for (auto mit=map_.begin(), mend=map_.end(); mit!=mend; mit++)
        {
            VoxelBlock &voxel_block = *mit.value();
            if ((mit.key().cast<double>() * voxel_size_ - origin).squaredNorm() > (max_distance2)) {
                removed.emplace_back(mit.key());
                if (voxel_block.mbBad) continue;
                Eigen::Vector3d center = voxel_block.centroid();
                Eigen::Vector3d normal = voxel_block.estimate().first;
                auto rot = Eigen::AngleAxisd(acos(z0.dot(normal)), z0.cross(normal).normalized());
        
                Eigen::Vector3d p1w = rot * p1 + center;
                Eigen::Vector3d p2w = rot * p2 + center;
                Eigen::Vector3d p3w = rot * p3 + center;
                Eigen::Vector3d p4w = rot * p4 + center;
                geometry_msgs::Point msgs_p1, msgs_p2, msgs_p3, msgs_p4;
                
                msgs_p1.x=p1w(0); msgs_p1.y=p1w(1); msgs_p1.z=p1w(2);
                msgs_p2.x=p2w(0); msgs_p2.y=p2w(1); msgs_p2.z=p2w(2);
                msgs_p3.x=p3w(0); msgs_p3.y=p3w(1); msgs_p3.z=p3w(2);
                msgs_p4.x=p4w(0); msgs_p4.y=p4w(1); msgs_p4.z=p4w(2);
                {
                    boost::mutex::scoped_lock lock(mMutexInactiveMap);
                    inactive_map_.points.push_back(msgs_p1);
                    inactive_map_.points.push_back(msgs_p2);
                    inactive_map_.points.push_back(msgs_p3);
                    inactive_map_.points.push_back(msgs_p3);
                    inactive_map_.points.push_back(msgs_p4);
                    inactive_map_.points.push_back(msgs_p1);
                }
            }
        }
    }
    boost::unique_lock<boost::shared_mutex> write_lock(mMutexMap);
    for (auto lit=removed.begin(),lend=removed.end(); lit!=lend; lit++) map_.erase(*lit);
}
void AdamMap::RemovePointsFarFromLocation()
{
    if (mvpKeyFrames.empty()) return;
    RemovePointsFarFromLocation(mvpKeyFrames.back()->GetPose().translation());
}

void AdamMap::visualize(visualization_msgs::Marker &planes)
{
    Eigen::Vector3d p1(0.5*voxel_size_,  0.5*voxel_size_, 0.0);
    Eigen::Vector3d p2(-0.5*voxel_size_, 0.5*voxel_size_, 0.0);
    Eigen::Vector3d p3(-0.5*voxel_size_, -0.5*voxel_size_, 0.0);
    Eigen::Vector3d p4(0.5*voxel_size_,  -0.5*voxel_size_, 0.0);
    Eigen::Vector3d z0(0.0, 0.0, 1.0);
    planes.points.clear();

    boost::shared_lock<boost::shared_mutex> read_lock(mMutexMap);
    for (auto mit=map_.begin(), mend=map_.end(); mit!=mend; mit++)
    {
        auto &voxel_block = *mit.value();
        if (voxel_block.mbBad) continue;
        Eigen::Vector3d center = voxel_block.centroid();
        Eigen::Vector3d normal = voxel_block.estimate().first;
        auto rot = Eigen::AngleAxisd(acos(z0.dot(normal)), z0.cross(normal).normalized());
        
        Eigen::Vector3d p1w = rot * p1 + center;
        Eigen::Vector3d p2w = rot * p2 + center;
        Eigen::Vector3d p3w = rot * p3 + center;
        Eigen::Vector3d p4w = rot * p4 + center;
        geometry_msgs::Point msgs_p1, msgs_p2, msgs_p3, msgs_p4;
        
        msgs_p1.x=p1w(0);
        msgs_p1.y=p1w(1);
        msgs_p1.z=p1w(2);
        msgs_p2.x=p2w(0);
        msgs_p2.y=p2w(1);
        msgs_p2.z=p2w(2);
        msgs_p3.x=p3w(0);
        msgs_p3.y=p3w(1);
        msgs_p3.z=p3w(2);
        msgs_p4.x=p4w(0);
        msgs_p4.y=p4w(1);
        msgs_p4.z=p4w(2);

        planes.points.push_back(msgs_p1);
        planes.points.push_back(msgs_p2);
        planes.points.push_back(msgs_p3);
        planes.points.push_back(msgs_p3);
        planes.points.push_back(msgs_p4);
        planes.points.push_back(msgs_p1);
    }
}
void AdamMap::visualize(ros::Publisher pub)
{
    boost::mutex::scoped_lock lock(mMutexInactiveMap);
    pub.publish(inactive_map_);
}

#endif
