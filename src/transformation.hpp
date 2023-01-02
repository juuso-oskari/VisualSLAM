
#ifndef VISUAL_SLAM_TRANSFORMATION
#define VISUAL_SLAM_TRANSFORMATION

#include "isometry3d.hpp"
#include "frame.hpp"
#include "point.hpp"
#include "helper_functions.hpp"

#include <iostream>
#include <vector>

#include <chrono>
#include <Eigen/Dense> 

#include <tuple>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>

class Transformation {
    public:
        //empty constructor
        Transformation(){}
        //base constructor
        Transformation(cv::Mat& T): T_(T){}
        //alternative constructor
        Transformation(cv::Mat R, cv::Mat t){
            T_(cv::Rect(0,0,3,3)) = R;
            T_(cv::Rect(0,3,1,3)) = t;
        }

        cv::Mat GetTransformation() {
            return T_;
        }

        Eigen::MatrixXd GetEigen(){
            Eigen::MatrixXd eigen_mat;
            cv::cv2eigen(T_, eigen_mat);
            return eigen_mat;
        }

        

        void SetTransformation(cv::Mat T){
            T_ = T;
        }

        void SetTransformation(Eigen::MatrixXd T){
            cv::eigen2cv(T, T_);
        }


        cv::Mat GetRotation() {
            cv::Mat R = (cv::Mat1d(3,3) <<  T_.at<double>(0,0), T_.at<double>(0,1), T_.at<double>(0,2),T_.at<double>(1,0), T_.at<double>(1,1), T_.at<double>(1,2),T_.at<double>(2,0), T_.at<double>(2,1), T_.at<double>(2,2));
            return R;    
        }

        cv::Mat GetTranslation() {
            cv::Mat t =  (cv::Mat1d(3,3) << T_.at<double>(0,3), T_.at<double>(1,3), T_.at<double>(2,3));
            return t; 
        }

        cv::Mat GetInverseTransformation() {
            return T_.inv();
        }
        // returns the fraction N_inlierPoints / N_Points, where N_inlierPoints: number of points fitting the estimated transformation, N_points
        double GetValidFraction(){
            double fraction = 0;
            for(int i = 0; i < inlierMask.rows; i++){
                //std::cout << (int)mask.at<uchar>(i) << std::endl;
                int mask_val = inlierMask.at<uchar>(i);
                if(mask_val == 1){
                    fraction += 1;
                }
            }
            fraction /= inlierMask.rows;
            return fraction;
        }
        
        cv::Mat GetCVMat() const{
            return T_;
        }

        friend Transformation operator*(const Transformation& op1, const Transformation& op2);


        virtual void Estimate(){
            std::cout << "Virtual Estimate() called, usually an error" << std::endl;
        }

    protected:
        cv::Mat T_ = cv::Mat1d(4,4);
        cv::Mat inlierMask; // needed to assess the success of transform estimation
};

Transformation operator*(const Transformation& op1, const Transformation& op2){
    Transformation t = Transformation(op1.GetCVMat(), op2.GetCVMat());
    return t;
}


class Essential : public Transformation {
    public:
    void Estimate(const cv::Mat& points1, const cv::Mat& points2, const cv::Mat& K){
        cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 2, inlierMask);    
        cv::Mat R; // Rotation
        cv::Mat t; // translation
        cv::Mat triangulated_points_cv(3, points1.rows, CV_64F); // 3D locations for inlier points estimated using triangulation and the poses recovered from essential transform
        cv::recoverPose(E, points1, points2, K, R, t, 50, inlierMask, triangulated_points_cv);
        Eigen::MatrixXd R_; // convert to eigen for transformation calculations
        Eigen::VectorXd t_;
        cv::cv2eigen(R, R_);
        cv::cv2eigen(t, t_);
        Eigen::MatrixXd pos = Isometry3d(R_, t_).matrix().inverse();
        cv::eigen2cv(pos, T_); 
        triangulatedPoints = triangulated_points_cv.t(); // transpose and store to private member
    }

    private:
        cv::Mat triangulatedPoints; // recoverPose also estimates 3D locations for inlier points

};

class PnP : public Transformation {
    public:
    void Estimate(cv::Mat matched_3d, cv::Mat curMatchedPoints, cv::Mat cameraIntrinsicsMatrix, cv::Mat DistCoefficients) {
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(matched_3d, curMatchedPoints, cameraIntrinsicsMatrix, DistCoefficients, rvec, tvec, false, 100, 8.0F, 0.9899999999999999911, inliers);
        T_ = transformMatrix(rvec,tvec);
        //store inlier indices to mask
        inlierMask = cv::Mat::zeros(curMatchedPoints.rows, 1, CV_64F);
        for(int i = 0; i < inliers.rows; i++){
            int inlier_idx = inliers.at<int>(i);
            inlierMask.at<int>(inlier_idx) = 1; // conversion to uchar from int
        }
    }
};

#endif