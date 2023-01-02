
#include "point.hpp"
#include <Eigen/Dense> 
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>



void TestPointClass() {
    int id_point = 100;
    cv::Mat location_3D = (cv::Mat1d(1,3) << 20.0, -10.0, 40.5); 
    // Create point object with id and 3D location
    Point3D* pt_object = new Point3D(id_point, location_3D);

    if (pt_object->GetID() == id_point) {
        std::cout << "ID point test succefully passed" << std::endl;
    }

    if (pt_object->Get3dPoint().at<double>(1,1) == location_3D.at<double>(1,1) && pt_object->Get3dPoint().at<double>(1,2) == location_3D.at<double>(1,2) && pt_object->Get3dPoint().at<double>(1,3) == location_3D.at<double>(1,3) ) {
        std::cout << "3D point test succefully passed" << std::endl;
    }


}

int TestFrameClass(){
    // init helper classes
    FeatureExtractor feature_extractor = FeatureExtractor();
    FeatureMatcher feature_matcher = FeatureMatcher();
    // start reading rgb images in data folder
    std::string path = "../data/rgbd_dataset_freiburg3_long_office_household/rgb";
    std::vector<std::filesystem::path> files_in_directory;
    std::copy(std::filesystem::directory_iterator(path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
    std::sort(files_in_directory.begin(), files_in_directory.end());
    std::vector<std::filesystem::path>::iterator input_video_it = files_in_directory.begin();
    // create a Frame object from the first frame
    cv::Mat image1, dispImg;
    image1 = cv::imread(*input_video_it);
    std::shared_ptr<Frame> prev_frame = std::make_shared<Frame>(image1, 0);
    prev_frame->process(feature_extractor);
    prev_frame->SetAsKeyFrame();
    prev_frame->AddPose(cv::Mat::eye(4,4,CV_64F)); // add Identity as initial pose
    // skip few frames and then read the next image (tests also the capabilities of the feature matcher)
    int skip = 0;
    while(skip < 50){
        input_video_it++;
        skip++;
    }
    cv::Mat image2;
    image2 = cv::imread(*input_video_it);
    std::shared_ptr<Frame> cur_frame = std::make_shared<Frame>(image2, 1);
    cur_frame->process(feature_extractor);
    std::vector<cv::DMatch> matches; cv::Mat preMatchedPoints; cv::Mat preMatchedFeatures; cv::Mat curMatchedPoints; cv::Mat curMatchedFeatures;
    std::tuple<std::vector<cv::DMatch>, cv::Mat , cv::Mat, cv::Mat , cv::Mat> match_info
            = Frame::Match2Frames(prev_frame, cur_frame, feature_matcher);
    // parse tuple to objects
    matches = std::get<0>(match_info); preMatchedPoints = std::get<1>(match_info); preMatchedFeatures = std::get<2>(match_info);
    curMatchedPoints = std::get<3>(match_info); curMatchedFeatures = std::get<4>(match_info);
    // draw matches
    cv::drawMatches(prev_frame->GetRGB(), prev_frame->GetKeyPointsAsVector(),
    cur_frame->GetRGB(), cur_frame->GetKeyPointsAsVector(), matches, dispImg);
    cv::imshow("Display Image", dispImg);
    cv::waitKey(0);
    std::cout << "Frame class unit test passed succesfully (visual inspection)" << std::endl;
    return 0;
}

int TestMapClass(){
    
}
