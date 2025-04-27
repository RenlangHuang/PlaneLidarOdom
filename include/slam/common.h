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



namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen



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
