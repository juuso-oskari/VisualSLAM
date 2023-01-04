#ifndef VISUAL_SLAM_MAIN
#define VISUAL_SLAM_MAIN

#include <string>
#include <iostream>
#include <filesystem>
// opencv imports
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "frame.hpp"
#include "helper_functions.hpp"
#include "isometry3d.hpp"
#include "map.hpp"
#include "point.hpp"
#include "screen.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

// {} defines scope for compound statement, statically declared variables (stored in stack memory) get destroyed when leaving the scope
// with new operator we allocate memory from heap

int main(int argc, char** argv )
{
    // This line is required for all Easy3D applications
    easy3d::initialize();
    // Prepare and configure the visualization
    easy3d::Viewer viewer("Visual SLAM");
    auto points = new easy3d::PointCloud;
    auto poses = new easy3d::PointCloud;
    auto pose_vectors = new easy3d::Graph;
    viewer.add_model(points);
    viewer.add_model(poses);
    viewer.add_model(pose_vectors);
    ConfigureModel(points, 5.0f, true, easy3d::vec4(0.1, 0.5, 1.0, 1.0));
    ConfigureModel(poses, 10.0f, false, easy3d::vec4(0.1, 1.0, 0.0, 1.0));
    ConfigureModel(pose_vectors, 2.0f, true, easy3d::vec4(0.1, 1.0, 0.0, 1.0));
    // The PointClouds object can now be used to edit the point cloud data at runtime.
    PointClouds clouds(points, poses, pose_vectors);
    // Start passing points and poses to the UI from a separate thread
    // Notice the lambda function captures `clouds` by reference, this
    // is so it can send the point and pose data to it.
    SpawnWorkerThread([&]() {
        bool do_scale_depth = true;
        // camera intrinsics matrix
        // double fx = 535.4; double fy = 539.2; double cx = 320.1; double cy = 247.6; // 
        double fx = 501.4757919305817; double fy = 501.4757919305817; double cx = 421.7953735163109; double cy = 167.65799492501083; // 
        // Define camera intrinsics matrix
        cv::Mat K  = (cv::Mat1d(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        cv::Mat I = (cv::Mat1d(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        cv::Mat D = (cv::Mat1d(4,1) << 0.0, 0.0, 0.0, 0.0); // no distortion
        double essTh = 3.0/K.at<double>(0,0);
        // start reading rgb images in data folder
        std::string path = "../data/rgbd_dataset_freiburg3_long_office_household/rgb";
        std::vector<std::filesystem::path> files_in_directory;
        std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
        std::sort(files_in_directory.begin(), files_in_directory.end());
        // running indexing for frames
        int id_frame = 0;
        int id_point = 0;
        // create feature extractor and matcher with default arguments
        FeatureExtractor feature_extractor = FeatureExtractor();
        FeatureMatcher feature_matcher = FeatureMatcher();
        Map global_map;
        std::vector<std::filesystem::path>::iterator image_file_iterator = files_in_directory.begin();
        // initialize map with first two good frames called keyframes, i.e. estimation of pose transform and point locations succeeds  
        std::cout << "INITIALIZING MAP" << std::endl;
        global_map.InitializeMap(image_file_iterator, id_frame, id_point, feature_extractor, feature_matcher, K);
        global_map.BundleAdjustement(false, do_scale_depth);
        // Whenever the map gets a new frame, it should send it to 
        // the point clouds using the methods in `clouds`.
        // Example:
        //     clouds.AddPointsMatUpdate(created_points, false);
        //     clouds.AddPointsMatUpdate(camera_locs, true);
        int last_kf_idx = id_frame-1;
        int iterations_count = 0;
        while(iterations_count < 10 && image_file_iterator != files_in_directory.end()){
            std::cout << "TRACKING" << std::endl;
            global_map.localTracking(image_file_iterator, id_frame, id_point, feature_extractor, feature_matcher, K, D, clouds, false);
            std::cout << "MAPPING" << std::endl;
            global_map.localMapping(id_frame, id_point, feature_extractor, feature_matcher, K, D, last_kf_idx);
            std::cout << "DOING BUNDLE ADJUSTEMENT" << std::endl;
            global_map.BundleAdjustement(false, false, false);
            // visualize all points
            std::vector<cv::Mat> created_points = global_map.GetAll3DPoints();
            std::vector<cv::Mat> camera_locs = global_map.GetAllCameraLocations();
            std::vector<cv::Mat> camera_poses = global_map.GetAllPoses();
            clouds.Clear();
            clouds.AddPointsMatUpdate(created_points, false);
            clouds.AddPointsMatUpdate(camera_locs, true);
            clouds.AddPoseAnglesMatUpdate(camera_poses);
            last_kf_idx = id_frame - 1 ; // update last keyframe index
            std::cout << "Passing points to viewer" << std::endl;
            iterations_count++;
        }
    });
    auto ret = viewer.run();
    // Cleanup
    delete points->renderer();
    delete poses->renderer();
    delete pose_vectors->renderer();
    delete points;
    delete poses;
    delete pose_vectors;
    // Exit code
    return ret;
}
#endif