/*
#include "../src/frame.hpp"

int main(int argc, char** argv )
{
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
*/