#pragma once
#define PCL_NO_PRECOMPILE

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <map>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sophus/se3.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


typedef Eigen::Matrix<float, 1, 64> Vector64;
typedef float array64[64];

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace pcl {
    struct KeyPoint {
        PCL_ADD_POINT4D;
        float unc;
        array64 desc;
        PCL_MAKE_ALIGNED_OPERATOR_NEW;
    } EIGEN_ALIGN16;
}
POINT_CLOUD_REGISTER_POINT_STRUCT (
    pcl::KeyPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, unc, unc)
    (array64, desc, desc)
)


class BaseMapPoint
{
public:
    BaseMapPoint() {}
    BaseMapPoint(BaseMapPoint &pt) : xyz(pt.xyz) {}
    BaseMapPoint(Eigen::Vector3d pos) : xyz(pos) {}
    BaseMapPoint(double x, double y, double z) : xyz(x,y,z) {}
    virtual Eigen::Vector3d GetPos() { return xyz; }

    double Distance(BaseMapPoint &point) {
        return (point.GetPos() - xyz).norm();
    }
    double Distance(const Eigen::Vector3d &point) {
        return (point - xyz).norm();
    }
    double SquaredDistance(BaseMapPoint &point) {
        return (point.GetPos() - xyz).squaredNorm();
    }
    double SquaredDistance(const Eigen::Vector3d &point) {
        return (point - xyz).squaredNorm();
    }
    virtual void Transform(Sophus::SE3d pose) {
        xyz = pose * xyz;
    }
    virtual void Transform(Eigen::Isometry3d pose) {
        xyz = pose * xyz;
    }
    static Eigen::MatrixXf load_weight(std::string file, int rows, int cols) {
        Eigen::MatrixXf data(rows, cols);
        std::ifstream data_file(file, std::ios::in | std::ios::binary);
        const size_t num_elements = rows * cols;

        std::vector<float> data_buffer(num_elements);
        data_file.read(reinterpret_cast<char*>(&data_buffer[0]), num_elements*sizeof(float));
        data = Eigen::MatrixXf::Map(data_buffer.data(), rows, cols);
        return data;
    }
    static void softmax(std::vector<float> &data) {
        float sum_data = 0;
        float max_data = *std::max_element(data.begin(), data.end());
        for(size_t i = 0; i < data.size(); i++) {
            data[i] = exp(data[i] - max_data);
            sum_data += data[i];
        }
        std::transform(data.cbegin(), data.cend(), data.begin(),
            [&](const auto &_data) { return _data / sum_data; });
    }

protected:
    Eigen::Vector3d xyz;
};


class TicToc
{
public:
    TicToc()
    {
        tic();
    }
    void tic()
    {
        start = std::chrono::system_clock::now();
    }
    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000; // milliseconds
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};


class AverageMeter
{
public:
    AverageMeter() {}
    void update(float x)
    {
        data = data * (float)(count++) + x;
        data = data / (float)count;
    }
    float read()
    {
        return data;
    }
private:
    float data = 0;
    int count = 0;
};

class SummaryBoard
{
public:
    SummaryBoard() {}
    void register_meter(std::string s)
    {
        meters.insert(std::make_pair(s,AverageMeter()));
    }
    void update(std::string s, float x)
    {
        if (meters.find(s) == meters.end())
        {
            meters.insert(std::make_pair(s,AverageMeter()));
        }
        meters[s].update(x);
    }
    float read(std::string s)
    {
        return meters[s].read();
    }
    void summarize()
    {
        printf("------\n");
        for (auto mit=meters.begin(), mend=meters.end(); mit!=mend; mit++)
        {
            printf("%s : %f\n", mit->first.c_str(), mit->second.read());
        }
        printf("------\n");
    }
private:
    std::map<std::string, AverageMeter> meters;
};


struct JacobianTuple {
    JacobianTuple() {
        JTJ.setZero();
        JTr.setZero();
    }
    JacobianTuple operator+(const JacobianTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }
    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};
struct JacobianTupleVector {
    JacobianTupleVector(size_t n) {
        JTJ.reserve(n);
        JTe.reserve(n);
        for (size_t i = 0; i < n; i++) {
            JTJ.push_back(Eigen::Matrix6d::Zero());
            JTe.push_back(Eigen::Vector6d::Zero());
        }
    }
    JacobianTupleVector operator+(const JacobianTupleVector &other) {
        for (size_t i = 0; i < this->JTJ.size(); i++) {
            this->JTJ[i] += other.JTJ[i];
            this->JTe[i] += other.JTe[i];
        }
        return *this;
    }
    std::vector<Eigen::Matrix6d> JTJ;
    std::vector<Eigen::Vector6d> JTe;
};

struct JacobianHessianTuple {
    JacobianHessianTuple(const int n) { // Gauss-Newton
        DTHD = Eigen::MatrixXd::Zero(n*6, n*6);
        DTJT = Eigen::MatrixXd::Zero(n*6, 1);
    }
    JacobianHessianTuple(const int n, double lambda) { // Levenberg-Marquardt
        DTHD = Eigen::MatrixXd::Identity(n*6, n*6) * lambda;
        DTJT = Eigen::MatrixXd::Zero(n*6, 1);
    }
    JacobianHessianTuple operator+(const JacobianHessianTuple &other) {
        this->DTHD += other.DTHD;
        this->DTJT += other.DTJT;
        return *this;
    }
    Eigen::MatrixXd DTHD;
    Eigen::VectorXd DTJT;
};


/*torch::Tensor Eigen2Tensor(Eigen::MatrixXd x)
{
    Eigen::MatrixXf y = x.cast<float>();
    return torch::from_blob(y.data(), {y.cols(),y.rows()}, torch::kFloat32).t();
}
Eigen::MatrixXd Tensor2Eigen(torch::Tensor t)
{
    Eigen::MatrixXf mat(t.size(1),t.size(0));
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(), mat.data());
    return mat.transpose().cast<double>();
}*/