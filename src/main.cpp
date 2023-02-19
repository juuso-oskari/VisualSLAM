#ifndef VISUAL_SLAM_MAIN
#define VISUAL_SLAM_MAIN

#include <yaml-cpp/yaml.h>
// load the yaml configuration file as global variable so all included files can see it and read it
YAML::Node config = YAML::LoadFile("../src/config.yaml");


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
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#include "viewer.hpp"
#include <thread>
#include <cnpy.h>
#include <fstream>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>


// {} defines scope for compound statement, statically declared variables (stored in stack memory) get destroyed when leaving the scope
// with new operator we allocate memory from heap

int main(int argc, char** argv )
{
    
    Viewer viewer;
    std::thread viewer_thread(&Viewer::Run, &viewer);
    bool do_scale_depth = true;
    // camera intrinsics matrix
    // double fx = 535.4; double fy = 539.2; double cx = 320.1; double cy = 247.6; // 
    //double fx = config["fx"].as<double>(); double fy = config["fy"].as<double>(); double cx = config["cx"].as<double>(); double cy = config["cy"].as<double>(); // 
    // Define camera intrinsics matrix
    //cv::Mat K  = (cv::Mat1d(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    cv::Mat K = parseCameraIntrinsics(config["K"].as<std::string>());
    cv::Mat I = (cv::Mat1d(3,3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat D = parseDistortionCoefficients(config["D"].as<std::string>());
    std::cout << D << std::endl;
    //cv::Mat D = (cv::Mat1d(4,1) << 0.0, 0.0, 0.0, 0.0); // no distortion
    double essTh = 3.0/K.at<double>(0,0);
    // start reading rgb images in data folder
    std::string path = config["dataset"].as<std::string>();
    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end());
    std::vector<std::filesystem::path>::iterator image_file_iterator = files_in_directory.begin();
    // running indexing for frames
    int id_frame = 0;
    int id_point = 0;
    // create feature extractor and matcher with default arguments
    FeatureExtractor feature_extractor = FeatureExtractor();
    FeatureMatcher feature_matcher = FeatureMatcher();
    Map global_map;
    // skip few initial frames
    for(int i=0; i<50; i++){
        image_file_iterator++;
    }
    // initialize map with first two good frames called keyframes, i.e. estimation of pose transform and point locations succeeds  
    std::cout << "INITIALIZING MAP" << std::endl;
    global_map.InitializeMap(image_file_iterator, id_frame, id_point, feature_extractor, feature_matcher, K, true);
    global_map.BundleAdjustement(false, K, do_scale_depth);
    int last_kf_idx = id_frame-1;
    int iterations_count = 0;
    while(iterations_count < config["max_iterations"].as<int>() && image_file_iterator != files_in_directory.end()){
        std::cout << "TRACKING" << std::endl;
        global_map.localTracking(image_file_iterator, id_frame, id_point, feature_extractor, feature_matcher, K, D, true, false);
        std::cout << "MAPPING" << std::endl;
        global_map.localMapping(id_frame, id_point, feature_extractor, feature_matcher, K, D, last_kf_idx, false, true);
        global_map.BundleAdjustement(false, K, false, false, 10);
        // visualize all points
        std::vector<cv::Mat> created_points = global_map.GetAll3DPoints();
        std::vector<cv::Mat> camera_locs = global_map.GetAllCameraLocations();
        std::vector<cv::Mat> camera_poses = global_map.GetAllPoses(true);
        // update viewer
        viewer.SetPoints(CvMatToEigenVector(created_points));
        viewer.SetPoses(CvMatToEigenIsometry(camera_poses));
        last_kf_idx = id_frame - 1 ; // update last keyframe index
        iterations_count++;
        std::cout << "ITERATION: " << iterations_count << std::endl;
    }
    // Save map into folder: images into folder/images, poses into folder/poses.txt, K into folder/K.txt
    global_map.saveMap(K);
    return 0;
}
#endif