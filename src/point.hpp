#ifndef VISUAL_SLAM_POINT
#define VISUAL_SLAM_POINT

#include <iostream>
#include <vector>
#include <Eigen/Dense> 
#include "frame.hpp"
#include <tuple>
#include <map>
#include <iostream>

/** @brief Point class is used to store point id, 3D point locations and corresponding frame information.
    @author Jere Knuutinen
    @date December 2022
*/

class Point3D{
public:
    /** Constructor 
        @param ID - Unique identifier of the point when added
        @param location_3d_ - 3D location of the point
    */
    Point3D(int ID, cv::Mat location_3d) : ID_(ID), location_3d_(location_3d) {
        //std::cout << "Point Base constructor called " << std::endl;
        ID_ = ID;
        location_3d_ = location_3d;
    }


    /** Copy constructor 
        @param p constant reference to point object 
    */
    // Point3D(const Point3D& p) {
    //     //std::cout << "Point Copy constructor called " << std::endl;
    //     ID_ = p.ID_;
    //     location_3d_ = p.location_3d_;
    //     frames_ = p.frames_;
    // }
    
    
    /** method operator= performs copy assignment
   * @param t constant reference to point object 
   * @returns reference to t
   */
    //Point3D& operator=(const Point3D& t)
    //{
        //std::cout << "Point Assignment operator called " << std::endl;
       // return *this;
    //}
    
    // destructor
    //~Point3D();

    /**
   * @brief method GetID returns point_id
   * @return int type corresponding to frame_id
   */
    int GetID() {
        return ID_;
    }
    
    /** Method GetFrame gets frame with frame id
   * @param frame_id int corresponding to frame id
   * @returns shared pointer corresponding to frame
   */
    std::shared_ptr<Frame> GetFrame(int frame_id) {
        if (frames_.find(frame_id) == frames_.end()) {
        // not found
        throw std::invalid_argument("Tried to accessing non-existing frame in point to fetch frame pointer");
        return nullptr;
        } else {
        // found
            auto it = frames_.find(frame_id);
            // return second element from map. ie. return value. not all values but first value that is frame pointer
            return std::get<0>(it->second);
        }
    }

    cv::Mat GetFrame2(int frame_id) {
        if (frames_.find(frame_id) == frames_.end()) {
        // not found
        throw std::invalid_argument("Tried to accessing non-existing frame in point to fetch image point (ver2)");
        cv::Mat ret;
        return ret;
        } else {
        // found
            auto it = frames_.find(frame_id);
            // return second element from map. ie. return value. not all values but second value that is image point
            return std::get<1>(it->second);
        }
    }


    /** Method GetImagePoint gets image point with frame_id
   * @param frame_id int corresponding to frame id
   * @returns cv::Mat image points
   */
    cv::Mat GetImagePoint(int frame_id) {
        if (frames_.find(frame_id) == frames_.end()) {
        // not found
        throw std::invalid_argument("Tried to access non-existing frame in point to fetch image point");
        cv::Mat ret;
        return ret;
        } else {
        // found
            auto it = frames_.find(frame_id);
            // return second element from map. ie. return value. not all values but second value that is image point
            return std::get<1>(it->second);
        }
    }

    
    std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> SubsetOfFrames(int frame_id) {
        std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> ret;
        for(auto it = frames_.begin(); it != frames_.end(); ++it) {
            if (frame_id == it->first) {
                ret[frame_id] = std::tuple(std::get<0>(it->second), std::get<1>(it->second), std::get<2>(it->second));
                return ret;
            }
        }
    }
    
    /** Method AddFrame adds frame to frames_ map container
   * @param frame std::shared_ptr corresponding to frame object
   * @param uv cv::Mat corresponding to image points
   * @returns cv::Mat corresponding to descriptors
   */
    void AddFrame(std::shared_ptr<Frame> frame, cv::Mat uv, cv::Mat descriptor) {
        frames_[frame->GetID()] = std::tuple(frame, uv, descriptor);
    }

    /** Method UpdatePoint updates 3d point location
   * @param uv cv::Mat corresponding to new 3d location
   */
    void UpdatePoint(cv::Mat new_location) {
        location_3d_ = new_location;
    }

    /** Method IsVisibleTo checks if the frame with frame_id sees this point
   * @param frame_id id corresponding to frame idetifier
   * @returns bool true if frame sees that point, else false
   */
    bool IsVisibleTo(int frame_id) {
        for(auto it = frames_.begin(); it != frames_.end(); ++it) {
            if (frame_id == it->first) {
                return true;
            }
        }
        return false;
    }

    /** Method GetImagePointAndFeature gets imagepoint and feature with frame id
   * @param frame_id id corresponding to frame idetifier
   * @returns std::tuple<cv::Mat, cv::Mat> corresponding to image point and descriptors
   */
    // get imagepoint and feature with frame id (1x)
    std::tuple<cv::Mat, cv::Mat> GetImagePointAndFeature(int frame_id) {
        if(frames_.find(frame_id)==frames_.end()){
            std::cout << "Unknown frame: " << std::to_string(frame_id) << " to point: " << std::to_string(ID_) << std::endl;
            std::tuple<cv::Mat, cv::Mat> ret;
            return ret; // return empty
        }
        std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat> temp = frames_[frame_id];
        std::tuple<cv::Mat, cv::Mat> ret = std::make_tuple(std::get<1>(temp), std::get<2>(temp));
        return ret;
    }
 
    /** Method Get3dPoint getter for getting 3d location
   * @returns cv::Mat corresponding to 3d point
   */
    cv::Mat Get3dPoint() {
        return location_3d_;
    }

    /** Method GetNVisibleFrames getter for getting number of frames that see the point
   * @returns cv::Mat corresponding to 3d point
   */
    int GetNVisibleFrames() {
        return frames_.size();
    }
    /** Method SetFrames sets sub set of frames
   * @param std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> map corresponding to frame information
   */
    void SetFrames(std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> subset_of_frames) {
        frames_ = subset_of_frames;
    }

    /** Method GetFrames getter for getting frames
   * @returns std::map<int, std::tuple<std::shared_ptr<Frame>
   */
    std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>>& GetFrames(){
        return frames_;
    }

    std::vector<int> GetAllFrameIDs(){
        std::vector<int> frame_ids;
        for(auto it : frames_){
            frame_ids.push_back(it.first);
        }
        
        return frame_ids;
    }


    /** Method IsBad for determining if point should be used
   * @returns bool if point is seen with less than 3 frames return true, else false
   */
    bool IsBad() {
        if (frames_.size() < 3) {
            return true;
        }
        else {
            return false;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Point3D& p);

private:
    int ID_; //!< unique ID that defines the point (Bool)
    cv::Mat location_3d_; //!< 3D location of thhe point (cv::Mat)
    // 3D location of thhe point
    // map(key=Frame_ID, values = frame, image_point, descriptrors)
    std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> frames_; //!< map of frames that see this  particular point object (std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>>) // map of frames that see this  particular point object
};

std::ostream& operator<<(std::ostream& os, const Point3D& p){
    os << "Point with ID: " << p.ID_ << ", 3d location: " << p.location_3d_ << ", And Point->Frame correspondences:\n";
    for(auto it2 = p.frames_.begin(); it2 != p.frames_.end(); ++it2){
        os << "Frame id: " << it2->first << std::flush << ", uv: " << std::get<1>(it2->second) << "\n" << std::flush;
    }
    return os;
};


#endif
