#ifndef VISUAL_SLAM_FRAME
#define VISUAL_SLAM_FRAME

#include <iostream>
#include <map>
#include "helper_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/xfeatures2d.hpp>

class FeatureExtractor{
public:
    FeatureExtractor(){};

    // takes video frame as input, outputs vector with keypoints as first element and corresponding descriptors as second element
    std::tuple<cv::Mat, cv::Mat> compute_features(const cv::Mat& img){ //std::tuple<cv::MatrixXd, cv::MatrixXd>
        std::vector<cv::KeyPoint> keypoints;
        detector->detect ( img,keypoints );
        cv::Mat descriptors;
        descriptor->compute ( img, keypoints, descriptors);
        /*
        cv::Mat output;
        cv::drawKeypoints(img, keypoints, output);
        cv::imwrite("../ORB_result.jpg", output);
        */
        return std::tuple(KeyPoint2Mat(keypoints), descriptors); 
    }

private:
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1500);
    //cv::Ptr<cv::SIFT> detector = cv::SIFT::create(1500);
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(1500);
    //cv::Ptr<cv::SIFT> descriptor = cv::SIFT::create(1500);
};

class FeatureMatcher{
public:
    FeatureMatcher(){
        matcher = cv::BFMatcher(cv::NORM_L2, false);
    };
    std::tuple<std::vector<cv::DMatch>, cv::Mat, cv::Mat, cv::Mat, cv::Mat>
    match_features(cv::Mat kp1, cv::Mat desc1, cv::Mat kp2, cv::Mat desc2, float ratio = 0.80){
        std::vector<std::vector< cv::DMatch > > rawMatches;
        //matcher.match(descriptors1, descriptors2, matches);
        matcher.knnMatch(desc1, desc2, rawMatches, 2);
        // perform Lowe's ratio test to get actual matches
        std::vector<cv::DMatch> matches;
        cv::Mat pts1;        
        cv::Mat pts2;
        cv::Mat ft1;
        cv::Mat ft2;
        for(auto it = rawMatches.begin(); it != rawMatches.end(); it++){
            if( (*it)[0].distance < ratio * (*it)[1].distance ){
                pts1.push_back( kp1.row((*it)[0].queryIdx) );
                pts2.push_back( kp2.row((*it)[0].trainIdx) );
                ft1.push_back( desc1.row((*it)[0].queryIdx) );
                ft2.push_back( desc2.row((*it)[0].trainIdx) );
                matches.push_back((*it)[0]);
            }
        }
        return std::tuple(matches, pts1, ft1, pts2, ft2);
    }
private:
    cv::BFMatcher matcher;

};

/** @brief Frame class is used to store extracted information about the video frames.
    
    @author Juuso Korhonen
    @date December 2022
    */


class Frame{
public:
    /** Empty constructor.
        */
    Frame(){};
    /** Constructor 
        @param rgb_img - video frame as cv::Mat, for example the output of cv::imread("frame.png")
        @param id - unique identifier of the frame when added to map
        */
    Frame(cv::Mat rgb_img, int id){
        //std::cout << "Base constructor called " << std::endl;
        rgb = rgb_img;
        ID = id;
        keyframe = false;

    }
    /** Constructor 
        @param rgb_path - path to video frame file
        @param id - unique identifier of the frame when added to map
        */
    Frame(std::string rgb_path, int id){
        //std::cout << "Base constructor called " << std::endl;
        rgb = cv::imread(rgb_path);
        ID = id;
        keyframe = false;

    }


    
//     /** Copy constructor 
//         @param f std::shared_ptr<Frame> smart shared pointer to Frame object
//         */
//     Frame(const std::shared_ptr<Frame> f){
//         //std::cout << "Copy constructor called " << std::endl;
//         rgb = f->rgb;
//         keypoints = f->keypoints;
//         features = f->features;
//         pose = f->pose;
//         ID = f->ID;
//         parents = f->parents;
//         keyframe = f->keyframe;

//     }


//     /** method operator= performs copy assignment
//    * @param t constant reference to Frame object 
//    * @returns reference to t
//    */
//     Frame& operator=(const Frame& t)
//     {
//         //std::cout << "Assignment operator called " << std::endl;
//         return *this;
//     }
    

    /** matches 2 Frame objects
   * @param prev_frame shared pointer to previous frame
   * @param cur_frame shared pointer to current frame
   * @param feature_matcher FeatureMatcher object to be used for feature matching
   * @returns tuple containing matching indices (for keypoints and descriptors) for previous frame (in col(0)) and current frame (in col(1)), matching keypoints in previous frame, matching descriptors in previous frame,
   * matching keypoints in current frame, matching descriptors in current frame
   */

    static std::tuple<std::vector<cv::DMatch>, cv::Mat, cv::Mat, cv::Mat, cv::Mat> Match2Frames(std::shared_ptr<Frame> prev_frame, std::shared_ptr<Frame> cur_frame, FeatureMatcher feature_matcher){
        return feature_matcher.match_features(prev_frame->GetKeyPoints(), prev_frame->GetFeatures(), cur_frame->GetKeyPoints(), cur_frame->GetFeatures());
    }


    /** method AddParent adds parent frame to the current frame
   * @param parent_frame_id int type frame id corresponding to the parent frame
   * @param transition cv::Mat type 4x4 transformation matrix corresponding to the mapping between the frames
   */
    void AddParent(int parent_frame_id, cv::Mat transition){
        parents.insert({parent_frame_id, transition});
    }





    /** method GetParentIDs returns all the parent ids
   * @return return std::vector<int> type array of frame ids
   */
    std::vector<int> GetParentIDs(){
        std::vector<int> keys;
        for(auto it = parents.begin(); it != parents.end(); it++) {
            keys.push_back(it->first);
            //std::cout << "Key: " << it->first << std::endl();
        }
        return keys;
    }

    /** method GetTransitionWithParentID returns 4x4 trasition matrix between the parent frame and the current frame
   * @return std::vector<int> type array of frame ids
   */
    cv::Mat GetTransitionWithParentID(int parent_id){
        return parents[parent_id];
    }


    /** method feature_extract extract features from rgb image with helper class FeatureExtractor object
   * @return std::tuple<cv::Mat, cv::Mat> type corresponding to image points and features
   */
    std::tuple<cv::Mat, cv::Mat>  feature_extract(cv::Mat rgb_img, FeatureExtractor feature_extractor){
        return feature_extractor.compute_features(rgb_img);
    }


    /** method process frame and return keypoints, features and the rgb image from where they were found
     * @param FeatureExtractor intance of class FeatureExtractor
     * @return std::tuple<cv::Mat, cv::Mat, cv::Mat> type corresponding to keypoints, features and the rgb image
   */ 
    std::tuple<cv::Mat, cv::Mat, cv::Mat> process_frame(FeatureExtractor feature_extractor){
        std::tuple<cv::Mat, cv::Mat> ft;
        ft = this->feature_extract(rgb, feature_extractor);
        SetKeyPoints(std::get<0>(ft)); // set private vars with setter
        SetFeatures(std::get<1>(ft));
        return std::tuple(std::get<0>(ft), std::get<1>(ft), rgb);
    }

    /**
    * @brief method process extracts features and processes the frame
    * @param FeatureExtractor intance of class FeatureExtractor
   */
    void process(FeatureExtractor feature_extractor){
        std::tuple<cv::Mat, cv::Mat> ft;
        ft = this->feature_extract(rgb, feature_extractor);
        SetKeyPoints(std::get<0>(ft)); // set private vars with setter
        SetFeatures(std::get<1>(ft));
    }



    /**
   * @brief method GetRGB is getter for the RGB image
   * @return cv::Mat type corresponding to RGB image
   */
    cv::Mat GetRGB() const{
        return rgb;
    }
    /**
   * @brief method SetRGB sets RGB image
   * @param new_rgb type of cv::Mat
   */
    void SetRGB(cv::Mat new_rgb){
        rgb = new_rgb;
    }

    /**
   * @brief method AddPose adds initial pose
   * @param new_rgb type of cv::Mat corresponding to 4x4 transormation matrix in the world frame
   */
    void AddPose(cv::Mat init_pose){
        pose = init_pose;
    }

    /**
   * @brief method UpdatePose  does the same as add pose, but for clarity different function name
   * @param new_rgb type of cv::Mat corresponding to 4x4 transormation matrix in the world frame
   */
    void UpdatePose(cv::Mat new_pose){
        pose = new_pose;
    }
    
    /**
   * @brief method GetPose is getter for the 4x4 transformation matrix
   * @return cv::Mat type corresponding to pose
   */
    cv::Mat GetPose() const{
        return pose;
    }
    /**
   * @brief method GetPose returns keyframe information
   * @return bool type corresponding to if the frame is keyframae
   */
    bool IsKeyFrame() const{
        return keyframe;
    }

    /**
   * @brief method SetAsKeyFrame is setter for the keyframe, set boolean flag to true if the frame is a keyframe
   */
    void SetAsKeyFrame(){
        keyframe = true;
    }

    /**
   * @brief method SetKeyPoints is setter for the image points
   * @param new_points type of cv::mat corresponding to the image points
   */
    void SetKeyPoints(cv::Mat new_points){
        keypoints = new_points;
    }

    /**
   * @brief method GetPoGetKeyPointse returns keypoints
   * @return cv::Mat type corresponding to image points
   */
    cv::Mat GetKeyPoints() const{
        return keypoints;
    }

    /**
   * @brief method GetKeyPointsAsVector returns keypoints as a std::vector, Only visualization needs them in this form
   * @return std::vector<cv::KeyPoint> type corresponding to image points
   */
    std::vector<cv::KeyPoint> GetKeyPointsAsVector() const{
        std::vector<cv::KeyPoint> vector_of_kp;
        for(int i = 0; i < keypoints.rows; i++){
            cv::KeyPoint kp;
            kp.pt.x = keypoints.at<double>(i,0);
            kp.pt.y = keypoints.at<double>(i,1);
            vector_of_kp.push_back(kp);
        }
        return vector_of_kp;
    }

    static std::vector<cv::KeyPoint> GetKeyPointsAsVector(cv::Mat mat_keypoints){
        std::vector<cv::KeyPoint> vector_of_kp;
        for(int i = 0; i < mat_keypoints.rows; i++){
            cv::KeyPoint kp;
            kp.pt.x = mat_keypoints.at<double>(i,0);
            kp.pt.y = mat_keypoints.at<double>(i,1);
            vector_of_kp.push_back(kp);
        }
        return vector_of_kp;
    }

    /**
   * @brief method SetFeatures is setter for the features
   * @param new_features type of cv::mat corresponding to the new features
   */
    void SetFeatures(cv::Mat new_features){
        features = new_features;
    }

    /**
   * @brief method SetKeyPoints Getter for the features
   * @return new_features type of cv::mat corresponding to the new features
   */
    cv::Mat GetFeatures() const{
        return features;
    }

    /**
   * @brief method GetID returns frame_id
   * @return int type corresponding to frame_id
   */
    int GetID() const{
        return ID;
    }
    /**
   * @brief method SetFeatures is setter for the frame_id
   * @param new_id
   */
    void AddID(int new_id){
        ID = new_id;
    }

    /**
   * @brief get camera center
   */

    cv::Mat GetCameraCenter(){
        return GetTranslation(pose);
    }

private:
    cv::Mat rgb; //!< rgb image stored in cv::Mat
    cv::Mat keypoints; //!< extracted keypoints (=imagepoints) stored in Nx2 cv::Mat
    cv::Mat features; //!< extracted descriptors for keypoints stored in Nxfeature_length cv::Mat
    cv::Mat pose; //!<  estimated camera pose during the frame
    int ID; //!<  unique identifier for the frame when stored in map
    std::map<int, cv::Mat> parents; //!<  std::map storing parents of this frame (useful when building a graph)
    bool keyframe = false; //!<  boolean flag indicating if the frame is considered keyframe
};

#endif
