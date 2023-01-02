#ifndef VISUAL_SLAM_ISOMETRY3D
#define VISUAL_SLAM_ISOMETRY3D

#include <iostream>
#include <vector>
#include <Eigen/Dense> 

#include <tuple>
#include <map>
using Eigen::MatrixXd;

class Isometry3d{
public:
    Isometry3d(Eigen::MatrixXd R, Eigen::VectorXd t) : R_(R), t_(t) {}

    Eigen::MatrixXd matrix() {
        Eigen::MatrixXd mat = Eigen::Matrix<double, 4,4>::Identity();
        //Eigen::Matrix <double, 4, 4> mat;
        mat.block<3,3>(0,0) = R_;
        mat.block<3,1>(0,3) = t_;
        return mat;
    }

    Isometry3d inverse() {
        return Isometry3d(this->R_.transpose(), -this->R_.transpose() * this->t_);
    }

    Eigen::MatrixXd orientation() {
        return R_;
    }

    Eigen::VectorXd position() {
        return t_;
    }

private:
    Eigen::MatrixXd R_;
    Eigen::VectorXd t_;
};

#endif