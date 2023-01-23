#ifndef VISUAL_SLAM_HELPER_FUNCTIONS
#define VISUAL_SLAM_HELPER_FUNCTIONS


#include "isometry3d.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense> 



#include <tuple>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef G2O_USE_VENDORED_CERES
#define G2O_USE_VENDORED_CERES
#endif

#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

int appendToFile(std::string filename, cv::Mat m) {
    std::ofstream out(filename, std::ios::app);
    //out << m << std::endl;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            out << m.at<double>(i,j);
            if( (i+1)*(j+1) < (m.cols*m.rows)){
                out << " ";
            }
            
        }
    }
    out << std::endl;
    out.close();
    return 0;
}



int appendToFile(std::string filename, cv::Mat m) {
    std::ofstream out(filename, std::ios::app);
    //out << m << std::endl;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            out << m.at<double>(i,j);
            if( (i+1)*(j+1) < (m.cols*m.rows)){
                out << " ";
            }
            
        }
    }
    out << std::endl;
    out.close();
    return 0;
}


cv::Mat parseDistortionCoefficients(std::string data_string){
    cv::Mat distortion_coefficients(5, 1, CV_64F);
    std::stringstream stream(data_string);
    for (int i = 0; i < 5; i++) {
        stream >> distortion_coefficients.at<double>(i);
    }
    return distortion_coefficients;
}

cv::Mat parseCameraIntrinsics(std::string data_string){
    cv::Mat distortion_coefficients(3, 3, CV_64F);
    std::stringstream stream(data_string);
    for (int i = 0; i < 9; i++) {
        stream >> distortion_coefficients.at<double>(i);
    }
    return distortion_coefficients;
}
    


/** @brief Function to read and modify frames
        * @param fp_iterator iterator that goes through png file paths
        * @returns img - the read image as cv::Mat
*/
cv::Mat readFrame(std::vector<std::filesystem::path>::iterator& fp_iterator, bool undistort = false){
    cv::Mat img;
    img = cv::imread(*fp_iterator);
    for(int i=0; i<1; i++){
        fp_iterator++;
    }
    if(config["do_undistort"].as<int>()==1){
        std::cout << "Do undistortion for image" << std::endl;
        cv::Mat undistorted_image;
        cv::Mat camera_matrix = parseCameraIntrinsics(config["K"].as<std::string>());
        cv::Mat distortion_coefficients = parseDistortionCoefficients(config["D"].as<std::string>());
        cv::undistort(img, undistorted_image, camera_matrix, distortion_coefficients);
        return undistorted_image;
    }

    return img;
}


/** @brief MakeHomogeneous make points homogenious, adds vector of ones
        * @param x - type cv::Mat corresponding to point
        * @returns cv::Mat - corresponding to homogenious vector
*/
cv::Mat MakeHomogeneous(cv::Mat x) {
    cv::Mat col_of_ones = cv::Mat::ones(x.rows, 1, CV_64F);
    cv::Mat ret;
    cv::hconcat(x, col_of_ones, ret);
    return ret;
}

cv::Mat AddRowOfOnes(cv::Mat x) {
    cv::Mat row_of_ones = cv::Mat::ones(1, x.cols, CV_64F);
    cv::Mat ret;
    cv::vconcat(x, row_of_ones, ret);
    return ret;
}


/** @brief CameraProjectionMatrix2 creates camera projection matrix
        * @param Pose - type cv::Mat, pose from world frame to camera frame
        * @param K - type cv::Mat, camera matrix
        * @returns cv::Mat - corresponding to projection matrix
*/
cv::Mat CameraProjectionMatrix2(cv::Mat Pose,cv::Mat K) {
    return K.t()*Pose(cv::Rect(0,0,4,3));
}

/** @brief GetRotation creates rotation matrix from transformation matrix
        * @param T_ - type cv::Mat, transformation matrix
        * @returns cv::Mat - corresponding to rotation
*/
cv::Mat GetRotation(cv::Mat T_) {
    cv::Mat R = (cv::Mat1d(3,3) <<  T_.at<double>(0,0), T_.at<double>(0,1), T_.at<double>(0,2),T_.at<double>(1,0), T_.at<double>(1,1), T_.at<double>(1,2),T_.at<double>(2,0), T_.at<double>(2,1), T_.at<double>(2,2));
    return R;    
}
/** @brief GetRotation creates translation matrix from transformation matrix
        * @param T_ - type cv::Mat, transformation matrix
        * @returns cv::Mat - corresponding to translation
*/
cv::Mat GetTranslation(cv::Mat T_) {
    cv::Mat t =  (cv::Mat1d(3,1) << T_.at<double>(0,3), T_.at<double>(1,3), T_.at<double>(2,3));
    return t; 
}
/*Taken partly from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/

/** @brief triangulate creates new 3D points
        * @param pose - type cv::Mat, Matrix from world to previous frame
        * @param pose - type cv::Mat, Matrix from world to cur frame
        * @param pts1 - type cv::Mat, image points in previous frame
        * @param pts2 - type cv::Mat, image points in cur frame
        * @param K - type cv::Mat, K camera matrix
        * @param inlierMask - type cv::Mat, inliermask
        * @returns cv::Mat - corresponding to triangluated points
*/
cv::Mat triangulate(cv::Mat pose1, cv::Mat pose2,cv::Mat pts1,cv::Mat pts2, cv::Mat K, cv::Mat& inlierMask) {
    cv::Mat ret;

    cv::Mat Rcw1 = GetRotation(pose1);
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = GetTranslation(pose1);;
    cv::Mat Tcw1(3,4,CV_64F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    
    //cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    cv::Mat Rcw2 = GetRotation(pose2);
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = GetTranslation(pose2);
    cv::Mat Tcw2(3,4,CV_64F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));
    
    double fx = K.at<double>(0,0); double fy = K.at<double>(1,1); double cx = K.at<double>(0,2); double cy = K.at<double>(1,2);
    double invfx = 1.0/fx; double invfy = 1.0/fy;

    double reproj_error = 0;
    for(int i = 0; i < pts1.rows; i++) { 
        cv::Mat xn1 = (cv::Mat_<double>(3,1) << (pts1.at<double>(i, 0)-cx)*invfx, (pts1.at<double>(i, 1)-cy)*invfy, 1.0);
        cv::Mat xn2 = (cv::Mat_<double>(3,1) << (pts2.at<double>(i, 0)-cx)*invfx, (pts2.at<double>(i, 1)-cy)*invfy, 1.0);
        cv::Mat x3D;
        cv::Mat A(4,4,CV_64F);
        A.row(0) = xn1.at<double>(0)*Tcw1.row(2)-Tcw1.row(0);
        A.row(1) = xn1.at<double>(1)*Tcw1.row(2)-Tcw1.row(1);
        A.row(2) = xn2.at<double>(0)*Tcw2.row(2)-Tcw2.row(0);
        A.row(3) = xn2.at<double>(1)*Tcw2.row(2)-Tcw2.row(1);
        cv::Mat w,u,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        x3D = vt.row(3).t();

        // Euclidean coordinates
        x3D = x3D.rowRange(0,3)/x3D.at<double>(3);
        
        cv::Mat x3Dt = x3D.t();
        ret.push_back(x3D.t());
        // Get camera centers
        
        // NOTICE! THIS IS BASICALLY COPIED FROM ORB-SLAM
        cv::Mat Ow1 = GetTranslation(pose1);
        cv::Mat Ow2 = GetTranslation(pose2);
        //Check triangulation in front of cameras
        double z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<double>(2);
        if(z1<=0){
            inlierMask.push_back((uchar)0);continue;
        }
            

        double z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<double>(2);
        if(z2<=0){
            inlierMask.push_back((uchar)0);continue;
        }
        
        //Check reprojection error in first keyframe
        const double &sigmaSquare1 = 20;//mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
        const double x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<double>(0);
        const double y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<double>(1);
        const double invz1 = 1.0/z1;

        double u1 = fx*x1*invz1+cx;
        double v1 = fy*y1*invz1+cy;
        double errX1 = u1 - pts1.at<double>(i, 0);
        double errY1 = v1 - pts1.at<double>(i, 1);

        //std::cout << "error : " << (errX1*errX1+errY1*errY1) << " threshold: " << 5.991*sigmaSquare1 << std::endl;
        reproj_error = reproj_error + (errX1*errX1+errY1*errY1);
        if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1){
            inlierMask.push_back((uchar)0);continue;
        }
            

        //Check reprojection error in second keyframe
        const double sigmaSquare2 = 20; //pKF2->mvLevelSigma2[kp2.octave];
        const double x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<double>(0);
        const double y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<double>(1);
        const double invz2 = 1.0/z2;

        double u2 = fx*x2*invz2+cx;
        double v2 = fy*y2*invz2+cy;
        double errX2 = u2 - pts2.at<double>(i, 0);
        double errY2 = v2 - pts2.at<double>(i, 1);
        if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2){
            inlierMask.push_back((uchar)0);continue;
        }
            

        //Check scale consistency
        cv::Mat normal1 = x3D-Ow1;
        double dist1 = cv::norm(normal1);

        cv::Mat normal2 = x3D-Ow2;
        double dist2 = cv::norm(normal2);

        if(dist1==0 || dist2==0){
            inlierMask.push_back((uchar)0);continue;
        }
            
        // NOTICE! THIS IS BASICALLY COPIED FROM ORB-SLAM UP TO HERE
         
        // /*const double ratioDist = dist2/dist1;
        // const double ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

        // /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
        //     continue;*/
        // if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
        //     continue;

        // */
        inlierMask.push_back((uchar)1);
    }


    //if((reproj_error / inlierMask.rows) > 1000){
      //  cv::waitKey(0);
    //}



    return ret;
}


/** @brief Returns indexes of kp1 rows that are not in kp2
    * @param pose - type cv::Mat, image points
    * @param pose - type cv::Mat, image points
    * @returns std::vector<int>  - returns index list
*/
std::vector<int> GetListDiff(cv::Mat kp1, cv::Mat kp2) {
    std::vector<int> idx_list;
    bool found = false;
    double eps = 1; // up to numerical instabilities
    for(int i_kp1 = 0; i_kp1 < kp1.rows; i_kp1++){ 
        found=false;
        for(int i_kp2 = 0; i_kp2 < kp2.rows; i_kp2++){ 
            if( (std::abs(kp1.at<double>(i_kp1,0) - kp2.at<double>(i_kp2,0)) < eps) && (std::abs((kp1.at<double>(i_kp1,1) - kp2.at<double>(i_kp2,1))) < eps) ) {
                found = true;
            }
        }
        if (found==false) {
            idx_list.push_back(i_kp1);
        }
    }
    return idx_list;
}
    

// get rows from m according to queryIdx in matches
/** @brief get rows from m according to queryIdx in matches
    * @param m - type cv::Mat
    * @param matches - type std::vector<cv::DMatch>
    * @returns cv::Mat 
*/
cv::Mat GetQueryMatches(cv::Mat m, std::vector<cv::DMatch> matches){
    cv::Mat matched_m;
    for(auto it = matches.begin(); it != matches.end(); it++){
        matched_m.push_back( m.row((*it).queryIdx));
    }
    return matched_m;
}

std::vector<int> GetQueryMatches(std::vector<int> point_ids, std::vector<cv::DMatch> matches){
    std::vector<int> matching_point_ids;
    for(auto it = matches.begin(); it != matches.end(); it++){
        matching_point_ids.push_back( point_ids[(*it).queryIdx]);
    }
    return matching_point_ids;
}

/** @brief gets image points
    * @param idx_list - type std::vector<int>, index list
    * @param image_points - type cv::Mat, image points
    * @returns cv::Mat - returns filtered image points
*/
cv::Mat GetImagePointsWithIdxList(std::vector<int> idx_list, cv::Mat image_points){
    cv::Mat new_image_points;
    for(auto it = idx_list.begin(); it != idx_list.end(); it++){
        //new_image_points.push_back(image_points.at<double>(*it));
        new_image_points.push_back(image_points.row(*it));
    }
    return new_image_points;
}
/** @brief gets descriptors 
    * @param idx_list - type std::vector<int>, index list
    * @param image_points - type cv::Mat, image points
    * @returns cv::Mat - returns filtered descriptors
*/
cv::Mat GetImageDescWithIdxList(std::vector<int> idx_list, cv::Mat image_points){
    cv::Mat new_image_points;
    for(auto it = idx_list.begin(); it != idx_list.end(); it++){
        new_image_points.push_back(image_points.row(*it));
    }
    return new_image_points;
}


/** @brief gets returns mask filtered results
    * @param inFrame - type cv::Mat
    * @param mask - type cv::Mat
    * @returns cv::Mat 
*/
cv::Mat MaskMat(cv::Mat inFrame, cv::Mat mask){
    cv::Mat outFrame;
    //inFrame.copyTo(outFrame, mask);
    //return outFrame;
    for(int i = 0; i < inFrame.rows; i++){
        //std::cout << (int)mask.at<uchar>(i) << std::endl;
        int mask_val = mask.at<uchar>(i);
        if(mask_val == 1){
            outFrame.push_back(inFrame.row(i));
        }
    }
    return outFrame;
    
}


// returns Nx2 cv::Mat

/** @brief converts keypoints to cv::Mat
    * @param keypoints - type std::vector<cv::KeyPoint>
    * @param mask - type cv::Mat
    * @returns cv::Mat Nx2
*/
cv::Mat KeyPoint2Mat(std::vector<cv::KeyPoint> keypoints){
    cv::Mat pointmatrix(keypoints.size(), 2, CV_64F);
    int row = 0;
    for (auto& kp: keypoints) {
        pointmatrix.at<double>(row, 0) = kp.pt.x;
        pointmatrix.at<double>(row, 1) = kp.pt.y;
        row++;
    }
    return pointmatrix;
}


/** @brief undistors imag
    @param keypoints - type std::vector<cv::KeyPoint>
    @param mask - type cv::Mat
    @returns cv::Mat Nx2
*/
cv::Mat KeyPoint2MatUndistord(std::vector<cv::KeyPoint> keypoints, cv::Mat cameraMatrix, cv::Mat distCoeffs, bool do_undistord = false){
    // convert to Point2f
    std::vector<cv::Point2f> points;
    cv::KeyPoint::convert(keypoints, points);

    std::vector<cv::Point2f> outputUndistortedPoints;
    if(do_undistord){
        cv::undistortPoints(points, outputUndistortedPoints, cameraMatrix, distCoeffs);
    }else{
        outputUndistortedPoints = points;
    }
    // flatten
    cv::Mat output = cv::Mat(outputUndistortedPoints.size(), 2, CV_64F, outputUndistortedPoints.data());
    return output;
}

/** @brief estimates the essential transformation from imagepoint correspondences in 2 frames
     * @param points1 image points in in frame 1
     * @param points2 image points in in frame 2
     * @param K camera intrinsics matrix
     * @param RelativePoseTransformation output array : pose transformation from camera pose in frame 1 to camera pose in frame 2
     * @param triangulatedPoints output array : triangulated 3D locations for points
     * @param inlierMask output array : mask indicating which points fit the estimated transformation and pass the cherilarity check for triangulation
   */ 
void EstimateEssential(const cv::Mat& points1, const cv::Mat& points2, const cv::Mat& K, cv::Mat& RelativePoseTransformation, cv::Mat& triangulatedPoints, cv::Mat& inlierMask){
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 2, inlierMask);    
    cv::Mat R; // Rotation
    cv::Mat t; // translation
    cv::Mat triangulated_points_cv(3, points1.rows, CV_64F); // 3D locations for inlier points estimated using triangulation and the poses recovered from essential transform
    cv::recoverPose(E, points1, points2, K, R, t, 50, inlierMask, triangulated_points_cv);
    Eigen::MatrixXd R_; // convert to eigen for transformation calculations
    Eigen::VectorXd t_;
    cv::cv2eigen(R, R_);
    cv::cv2eigen(t, t_);
    Eigen::MatrixXd pos = Isometry3d(R_, t_).inverse().matrix();
    cv::eigen2cv(pos, RelativePoseTransformation);
    //triangulatedPoints = triangulated_points_cv.t(); // transpose and return
    // make euclidean
    for(int i=0; i < triangulated_points_cv.cols; i++){
        cv::Mat x3D = triangulated_points_cv.col(i);
        triangulatedPoints.push_back( (x3D.rowRange(0,3)/x3D.at<double>(3)).t() );
    }

}
 
cv::Mat segment(cv::Mat mat, int start_idx, int end_idx){
    cv::Mat segmented_mat;
    for(int i = 0; i < mat.cols; i++){
        if(i>=start_idx && i<end_idx){
            segmented_mat.push_back(mat.col(i));
        }
    }
    return segmented_mat.t();
}

/** @brief creates transformation matrix from solvePnP results
     * @param rvec - Type cv::Mat 
     * @param tvec - Type cv::Mat
     * @returns cv::Mat - Transformation matrix
*/ 

cv::Mat transformMatrix(cv::Mat rvec, cv::Mat tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat T_temp;
    cv::hconcat(R,tvec,T_temp); // horizontal concatenation
    cv::Mat z  = (cv::Mat1d(1,4) << 0.0, 0.0, 0.0, 1.0);
    cv::Mat T;
    cv::vconcat(T_temp,z,T); // vertical   concatenation
    return T;
}



/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
g2o::SE3Quat toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<double>(0,0), cvT.at<double>(0,1), cvT.at<double>(0,2),
         cvT.at<double>(1,0), cvT.at<double>(1,1), cvT.at<double>(1,2),
         cvT.at<double>(2,0), cvT.at<double>(2,1), cvT.at<double>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<double>(0,3), cvT.at<double>(1,3), cvT.at<double>(2,3));

    return g2o::SE3Quat(R,t);
}
/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<double>(0), cvVector.at<double>(1), cvVector.at<double>(2);

    return v;
}


/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_64F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<double>(i,j)=m(i,j);

    return cvMat.clone();
}
/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
cv::Mat toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}
/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_64F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<double>(i,j)=m(i,j);

    return cvMat.clone();
}
/*Taken from Project ORB_SLAM2 (https://github.com/raulmur/ORB_SLAM2)*/
cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_64F);
    for(int i=0;i<3;i++)
            cvMat.at<double>(i)=m(i);

    return cvMat.clone();
}



/** @brief depth normalizes trasnlation vector
     * @param P - Type cv::Mat
     * @param median_depth - Type double
     * @returns cv::Mat - depth normalized result
*/ 
cv::Mat NormalizeTranslation(cv::Mat P, double median_depth){
    P.at<double>(0,3) = P.at<double>(0,3)/median_depth;
    P.at<double>(1,3) = P.at<double>(1,3)/median_depth;
    P.at<double>(2,3) = P.at<double>(2,3)/median_depth;
    return P;
}


/** @brief Convers points to homogenious
     * @param points3D - Type cv::Mat
     * @returns cv::Mat 
*/ 
cv::Mat Points2Homogeneous(cv::Mat points3D){
    cv::Mat points3D_euclidean;
    for(int i=0; i < points3D.rows; i++){
        cv::Mat x3D = points3D.row(i).t();
        x3D = x3D.rowRange(0,3)/x3D.at<double>(3);
        points3D_euclidean.push_back(x3D.t());
    }
    return points3D_euclidean;
}


std::vector<Eigen::Vector3d> CvMatToEigenVector(const std::vector<cv::Mat>& cv_points) {
  std::vector<Eigen::Vector3d> eigen_points;
  eigen_points.reserve(cv_points.size());
  for (const cv::Mat& cv_point : cv_points) {
    eigen_points.emplace_back(cv_point.at<double>(0), cv_point.at<double>(1), cv_point.at<double>(2));
  }
  return eigen_points;
}


std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> CvMatToEigenIsometry(const std::vector<cv::Mat>& cv_poses) {
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> eigen_poses;
  eigen_poses.reserve(cv_poses.size());
  for (const cv::Mat& cv_pose : cv_poses) {
    Eigen::Isometry3d eigen_pose;
    eigen_pose.matrix() << cv_pose.at<double>(0,0), cv_pose.at<double>(0,1), cv_pose.at<double>(0,2), cv_pose.at<double>(0,3),
                           cv_pose.at<double>(1,0), cv_pose.at<double>(1,1), cv_pose.at<double>(1,2), cv_pose.at<double>(1,3),
                           cv_pose.at<double>(2,0), cv_pose.at<double>(2,1), cv_pose.at<double>(2,2), cv_pose.at<double>(2,3),
                           cv_pose.at<double>(3,0), cv_pose.at<double>(3,1), cv_pose.at<double>(3,2), cv_pose.at<double>(3,3);
    eigen_poses.emplace_back(eigen_pose);
  }
  return eigen_poses;
}



/** @brief Convers points to homogenious
     * @param pose1 - 4x4 transformation matrix representing first pose
     * @param pose2 - 4x4 transformation matrix representing second pose
     * @returns pose error as double 
*/ 
double CalculatePoseDifference(cv::Mat pose1, cv::Mat pose2){
    // Extract translation and rotation from transformation matrices
    cv::Vec3d t1, t2;
    cv::Mat R1, R2;
    std::cout << "Piippii" << std::endl;
    cv::decomposeProjectionMatrix(pose1, R1, t1, cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());
    cv::decomposeProjectionMatrix(pose2, R2, t2, cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray());
    std::cout << "Juu" << std::endl;
    // Calculate translation error as Euclidean distance between translations
    double trans_error = cv::norm(t1, t2, cv::NORM_L2);

    // Calculate rotation error using angle between rotations
    cv::Mat R_error;
    cv::Rodrigues(R1.t() * R2, R_error);
    double rot_error = cv::norm(R_error, cv::NORM_L2);

    // Calculate overall pose error as weighted sum of translation and rotation errors
    double alpha = 0.5;  // Weight for translation error
    double beta = 1.0 - alpha;  // Weight for rotation error
    double pose_error = alpha * trans_error + beta * rot_error;
    return pose_error;
}


cv::Mat get3DPoints(const cv::Mat& depth_map, const cv::Mat& intrinsics) {
    // Get the dimensions of the depth map
    int rows = depth_map.rows;
    int cols = depth_map.cols;

    // Create a vector to store the 3D points
    cv::Mat points3D;

    // Iterate over all the pixels in the depth map
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // Get the depth value for this pixel
            double depth = depth_map.at<double>(y, x);

            // Calculate the 3D point using the formula:
            // 3D location = depth * (inverse of intrinsic matrix) * (pixel location in 2D)
            cv::Mat homogeneous_pixel_loc = (cv::Mat_<double>(3,1) << x, y, 1);
            cv::Mat point3D = depth * (intrinsics.inv()) * homogeneous_pixel_loc;

            // Add the 3D point to the vector
            points3D.push_back(point3D.t());
        }
    }

    // Return the vector of 3D points
    return points3D;
}

#endif
