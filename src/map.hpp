#ifndef VISUAL_SLAM_MAP
#define VISUAL_SLAM_MAP

#include "screen.hpp"
#include "isometry3d.hpp"
#include "frame.hpp"
#include "point.hpp"
#include "helper_functions.hpp"
#include "transformation.hpp"

#include <iostream>
#include <vector>

#include <chrono>
#include <Eigen/Dense> 

#include <tuple>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/videoio.hpp>

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

//#include "helper_functions.hpp"

/** @brief Map class is used to store Frame and Point3D objects.
    
    @author Juuso Korhonen, Jere Knuutinen
    @date December 2022
    */


class Map {
    public:
    /** @brief Initializes map based on first two keyframes from video (first being the 1st frame and 2nd is being found)
        @param input_video_it - iterator for going through the video frames (image file paths)
        @param id_frame - running indexing for Frame objects to be stored in the map
        @param id_point - running indexing for Point3D objects to be stored in the map
        @param feature_extractor - an instance of FeatureExtractor class, used for extracting information from the video frames
        @param feature_matcher - an instance of FeatureMatcher class, used for matching keypoints and features of two video frames
        @param cameraIntrinsicsMatrix - stores camera parameters in matrix form
        @param visualize - flag to tell if should visualize the matching process with opencv
        */
        void InitializeMap(std::vector<std::filesystem::path>::iterator& input_video_it, int& id_frame, int& id_point, FeatureExtractor feature_extractor, FeatureMatcher feature_matcher, cv::Mat cameraIntrinsicsMatrix, bool visualize = false){
            // store first frame as prev_frame
            cv::Mat img, dispImg;
            img = readFrame(input_video_it);
            //std::shared_ptr<Frame> prev_frame(new Frame(img, id_frame)); // create frame object out of image
            std::shared_ptr<Frame> prev_frame = std::make_shared<Frame>(img, id_frame);
            prev_frame->process(feature_extractor);
            prev_frame->SetAsKeyFrame();
            prev_frame->AddPose(cv::Mat::eye(4,4,CV_64F)); // add Identity as initial pose
            (*this).AddFrame(id_frame, prev_frame); // add to map
            id_frame++;
            // read next image
            img = readFrame(input_video_it);
            while(!img.empty()){
                //std::shared_ptr<Frame> cur_frame(new Frame(image, id_frame)); // create frame object out of image
                std::shared_ptr<Frame> cur_frame = std::make_shared<Frame>(img, id_frame);
                img = cv::imread(*input_video_it);
                input_video_it++;
                cur_frame->process(feature_extractor);
                std::vector<cv::DMatch> matches; cv::Mat preMatchedPoints; cv::Mat preMatchedFeatures; cv::Mat curMatchedPoints; cv::Mat curMatchedFeatures;
                std::tuple<std::vector<cv::DMatch>, cv::Mat , cv::Mat, cv::Mat , cv::Mat> match_info
                     = Frame::Match2Frames(prev_frame, cur_frame, feature_matcher);
                // parse tuple to objects
                matches = std::get<0>(match_info); preMatchedPoints = std::get<1>(match_info); preMatchedFeatures = std::get<2>(match_info);
                curMatchedPoints = std::get<3>(match_info); curMatchedFeatures = std::get<4>(match_info);
                // draw matches
                if(visualize){
                    cv::drawMatches(prev_frame->GetRGB(), prev_frame->GetKeyPointsAsVector(),
                    cur_frame->GetRGB(), cur_frame->GetKeyPointsAsVector(), matches, dispImg);
                    cv::imshow("Display Image", dispImg);
                    cv::waitKey(1);
                }
                if(matches.size() < 100){
                    ////std::cout << "Too few matches for map initialization, continuing to next frame" << std::endl;
                    continue;
                }
                //Essential transformation = Essential();               
                //transformation.Estimate(preMatchedPoints, curMatchedPoints, cameraIntrinsicsMatrix);
                ////std::cout << "Essential valid fraction: " << transformation.GetValidFraction() << std::endl;

                cv::Mat inlierMask;
                cv::Mat RelativePoseTransformation, TriangulatedPoints;
                EstimateEssential(preMatchedPoints, curMatchedPoints, cameraIntrinsicsMatrix, RelativePoseTransformation, TriangulatedPoints, inlierMask);
                if(cv::sum(inlierMask)[0]/preMatchedPoints.rows < 0.9 ){
                    ////std::cout << "Too few inliers " << cv::sum(inlierMask)[0] << "/" << preMatchedPoints.rows << " for map initialization, continuing to next frame" << std::endl;
                    continue;
                }
                
                // new keyframe is found
                cur_frame->SetAsKeyFrame();
                cv::Mat cur_pose = RelativePoseTransformation; // shortcut as previous pose is identity, inverse because opencv treats transformation as "where the points move" instead of "where the camera moves"
                // Adds cur frame to map with estimated pose, parent frame, and relative pose transformation between parent and frame

                (*this).AddPoseNode(id_frame, cur_frame, cur_pose, id_frame - 1, RelativePoseTransformation);
                id_frame++;
                // get inliers and turn to eigen matrices 
                (*this).AddPoints3D(id_point, TriangulatedPoints, prev_frame, preMatchedPoints, preMatchedFeatures, cur_frame, curMatchedPoints, curMatchedFeatures, inlierMask);
                // at end of loop
                break;
            }
        }

        /** @brief Tracks map points in consecutive video frames and estimates poses using PnP-algorithm. Breaks when a new keyframe is found:
        *  - 1. at least 20 frames has passed or current frame tracks less than 80 map points
        *  - 2. The map points tracked are fewer than 90% of the map points seen by the last key frame
        @param input_video_it - iterator for going through the video frames (image file paths)
        @param id_frame - running indexing for Frame objects to be stored in the map as reference, gets increased inside the map
        @param id_point - running indexing for Point3D objects to be stored in the map as reference, gets increased inside the map
        @param feature_extractor - an instance of FeatureExtractor class, used for extracting information from the video frames
        @param feature_matcher - an instance of FeatureMatcher class, used for matching keypoints and features of two video frames
        @param cameraIntrinsicsMatrix - stores camera parameters in matrix form
        @param DistCoefficients - stores camera distortion coefficients in vector form
        @param visualize - flag to tell if should visualize the matching process with opencv
        @param verbose_optimization - flag to tell if we should print details of the motion only optimization
        */

        void localTracking(std::vector<std::filesystem::path>::iterator& input_video_it, int& id_frame, int& id_point, FeatureExtractor feature_extractor, FeatureMatcher feature_matcher, cv::Mat cameraIntrinsicsMatrix, cv::Mat DistCoefficients, bool visualize = true, bool verbose_optimization = false){
            //Get the map points that the last keyframe sees
            int lastkeyframe_idx = id_frame-1;
            std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> map_points = GetImagePointsWithFrameID(lastkeyframe_idx); // get information of the points the last keyframe sees
            // start tracking
            cv::Mat image;
            image = readFrame(input_video_it);
            int trackFrameCount = 0;
            cv::Mat rvec, tvec;
            while(!image.empty()){
                // create Frame object from video frame and increase videoframe iterator
                //std::shared_ptr<Frame> cur_frame(new Frame(image, id_frame)); // create frame object out of image
                std::shared_ptr<Frame> cur_frame = std::make_shared<Frame>(image, id_frame);
                image = readFrame(input_video_it);
                cur_frame->process(feature_extractor);
                // pass imagepoints of map points (last keyframe), descriptors of map points (last keyframe), current frame imagepoints, and current frame for feature matching
                std::vector<cv::DMatch> matches; cv::Mat preMatchedPoints; cv::Mat preMatchedFeatures; cv::Mat curMatchedPoints; cv::Mat curMatchedFeatures;
                std::tuple<std::vector<cv::DMatch>, cv::Mat, cv::Mat, cv::Mat, cv::Mat> match_info = feature_matcher.match_features(std::get<0>(map_points), std::get<1>(map_points), cur_frame->GetKeyPoints(), cur_frame->GetFeatures());
                // parse tuple to objects
                matches = std::get<0>(match_info); preMatchedPoints = std::get<1>(match_info); preMatchedFeatures = std::get<2>(match_info); curMatchedPoints = std::get<3>(match_info); curMatchedFeatures = std::get<4>(match_info);
                // get 3d locations of matching imagepoints and corresponding point ids
                // matched_3d are 3d point locations of those map points that we are able to match in the current frame
                cv::Mat matched_3d = GetQueryMatches(std::get<2>(map_points), matches); 
                //std::cout << "TRACKING " << matched_3d.rows << " POINTS" << std::endl;
                // corresponding_point_ids are point ids of those map points that we are able to match in the current frame
                std::vector<int> corresponding_point_ids = GetQueryMatches(std::get<3>(map_points), matches);
                
                cv::Mat inliers;
                // Get last keyframe rotation vec and translation vec as initial guesses for solvepnp
                cv::Mat last_kf_pose = (GetFrame(id_frame-1)->GetPose());
                cv::Mat tvec = GetTranslation(last_kf_pose);
                cv::Mat rvec;
                cv::Rodrigues(GetRotation((last_kf_pose)), rvec);

                std::cout << "Initial guess for tvec: " << tvec << std::endl;
                //cv::solvePnPRansac(matched_3d, curMatchedPoints, cameraIntrinsicsMatrix, DistCoefficients, rvec, tvec, false, 200, 3.0F, 0.95, inliers);
                cv::solvePnPRansac(matched_3d, curMatchedPoints, cameraIntrinsicsMatrix, DistCoefficients, rvec, tvec, true, 300, 4.0F, 0.99, inliers);

                std::cout << "Optimized tvec: " << tvec << std::endl;

                if(inliers.rows<10){
                    continue;
                }
                std::cout << "Inliers passing solvePnPRansack: " << inliers.rows << "/" << curMatchedPoints.rows << std::endl;
                cv::Mat T = transformMatrix(rvec,tvec);
                cv::Mat W_T_curr = T.inv(); // From w to curr frame W_T_curr
                //PnP transformation = PnP();
                //transformation.Estimate(matched_3d, curMatchedPoints, cameraIntrinsicsMatrix, DistCoefficients);
                cv::Mat prev_T_W = (GetFrame(id_frame-1)->GetPose()).inv(); // From prev to world frame W_T_curr
                cv::Mat Relative_pose_trans = prev_T_W * W_T_curr;
                AddParentAndPose(id_frame-1, id_frame, cur_frame, Relative_pose_trans, W_T_curr);
                id_frame++;
                AddPointToFrameCorrespondances(corresponding_point_ids, curMatchedPoints, curMatchedFeatures, cur_frame, inliers);
                BundleAdjustement(true, false, verbose_optimization); // Do motion only (=points are fixed) bundleadjustement by setting tracking to true
                if(visualize){
                    cv::Mat dispImg;
                    cv::drawMatches(GetFrame(lastkeyframe_idx)->GetRGB(), Frame::GetKeyPointsAsVector(std::get<0>(map_points)), cur_frame->GetRGB(), cur_frame->GetKeyPointsAsVector(), matches, dispImg);
                    cv::imshow("Display Image", dispImg);
                    cv::waitKey(1);
                }
                // Check if current frame is a key frame:
                // 1. at least 20 frames has passed or current frame tracks less than 80 map points
                // 2. The map points tracked are fewer than 90% of the map points seen by the last key frame
                std::cout << "Inliers / map points seen by last kf: "<< ((double)inliers.rows) << "/" << ((double)std::get<0>(map_points).rows) << std::endl;
                std::cout << ((double)inliers.rows) / ((double)std::get<0>(map_points).rows) << std::endl;
                if( (trackFrameCount > 15 ||  inliers.rows < 200) && ( (((double)inliers.rows) / ((double)std::get<0>(map_points).rows)) < 0.9) ){ // || ( (((double)inliers.rows) / ((double)std::get<0>(map_points).rows)) < 0.9) ) { //|| (inliers.rows / std::get<0>(map_points).rows < 0.9)){
                //if( (trackFrameCount > 15 && inliers.rows < 120)  ){
                    std::cout<<"New keyframe found" << std::endl;
                    break;
                }
                trackFrameCount++;
                // visualize all points
                //std::vector<cv::Mat> created_points = GetAll3DPoints();
                //std::vector<cv::Mat> camera_locs = GetAllCameraLocations();
                
            }
            
        }

        /** @brief Does mapping using the last keyframe and new keyframe: Tries to find new matches between their descriptors and triangulate 3d locations for matches using their estimated poses
        @param id_frame - running indexing for Frame objects to be stored in the map as reference, gets increased inside the map
        @param id_point - running indexing for Point3D objects to be stored in the map as reference, gets increased inside the map
        @param feature_extractor - an instance of FeatureExtractor class, used for extracting information from the video frames
        @param feature_matcher - an instance of FeatureMatcher class, used for matching keypoints and features of two video frames
        @param cameraIntrinsicsMatrix - stores camera parameters in matrix form
        @param DistCoefficients - stores camera distortion coefficients in vector form
        @param last_key_frame_id - index of the last keyframe (store before tracking)
        @param visualize - flag to tell if should visualize the matching process with opencv
        @param verbose_optimization - flag to tell if we should print details of the motion only optimization
        */

        void localMapping(int& id_frame, int& id_point, FeatureExtractor feature_extractor, FeatureMatcher feature_matcher, cv::Mat cameraIntrinsicsMatrix, cv::Mat DistCoefficients,int& last_key_frame_id, bool visualize = false){
            GetFrame(id_frame-1)->SetAsKeyFrame(); // Set lastly added frame to be a new keyframe
            //# Get last keyframe pose from global map (from world to previous keyframe)
            cv::Mat W_T_prev_key = GetFrame(last_key_frame_id)->GetPose();
            cv::Mat prev_key_T_W = W_T_prev_key.inv(); // Get pose from previous keyframe to world
            cv::Mat W_T_cur_key = GetFrame(id_frame-1)->GetPose(); // Get pose from W to curr keyframe
            // (int parent_frame_id, cv::Mat transition)
            GetFrame(id_frame-1)->AddParent(last_key_frame_id, prev_key_T_W*W_T_cur_key);
            cv::Mat image_points_already_in_map = std::get<0>(GetImagePointsWithFrameID(last_key_frame_id));
            cv::Mat kp1 = GetFrame(last_key_frame_id)->GetKeyPoints(); // this contains all the image points (matched and unmatched)
            cv::Mat desc1 = GetFrame(last_key_frame_id)->GetFeatures(); 
            std::vector<int> idx_list = GetListDiff(kp1, image_points_already_in_map); // THIS DOES NOT WORK PERFECTLY CORREC; BUT SUFFICIENT FOR NOW
            // GetImagePointsWithIdxList
            cv::Mat unmatched_kp1 = GetImagePointsWithIdxList(idx_list, kp1);
            cv::Mat unmatched_desc1 = GetImageDescWithIdxList(idx_list, desc1);
            std::tuple<std::vector<cv::DMatch>, cv::Mat, cv::Mat, cv::Mat, cv::Mat> match_info = feature_matcher.match_features(unmatched_kp1, unmatched_desc1, GetFrame(id_frame-1)->GetKeyPoints(), GetFrame(id_frame-1)->GetFeatures());
            std::vector<cv::DMatch> matches; cv::Mat last_keyframe_points; cv::Mat last_keyframe_features; cv::Mat cur_keyframe_points; cv::Mat cur_keyframe_features;
            matches = std::get<0>(match_info); last_keyframe_points = std::get<1>(match_info); last_keyframe_features = std::get<2>(match_info); cur_keyframe_points = std::get<3>(match_info); cur_keyframe_features = std::get<4>(match_info);
            if(visualize){
                cv::Mat dispImg;
                cv::drawMatches(GetFrame(last_key_frame_id)->GetRGB(), Frame::GetKeyPointsAsVector(unmatched_kp1), GetFrame(id_frame-1)->GetRGB(), GetFrame(id_frame-1)->GetKeyPointsAsVector(), matches, dispImg);
                cv::imshow("Display Image", dispImg);
                cv::waitKey(1);
            }
            cv::Mat Proj1 = CameraProjectionMatrix2(GetFrame(last_key_frame_id)->GetPose(), cameraIntrinsicsMatrix);
            cv::Mat Proj2 = CameraProjectionMatrix2(GetFrame(id_frame-1)->GetPose(), cameraIntrinsicsMatrix);
            cv::Mat inlierMask;
            cv::Mat new_triagulated_points = triangulate(GetFrame(last_key_frame_id)->GetPose(), GetFrame(id_frame-1)->GetPose(), last_keyframe_points, cur_keyframe_points, cameraIntrinsicsMatrix, inlierMask);
            
            ////std::cout << "New triangulated points: " << new_triagulated_points.rows << "x" <<  new_triagulated_points.cols << std::endl;
            std::cout << "ADDING " << cv::sum(inlierMask)[0] << " NEW MAP POINTS" << std::endl;
            // cleanup bad points from map (seen by less than 3 frames)
            CleanUpBadPoints();

            AddPoints3D(id_point, new_triagulated_points, GetFrame(last_key_frame_id), last_keyframe_points, last_keyframe_features, GetFrame(id_frame-1), cur_keyframe_points, cur_keyframe_features, inlierMask);
        }

        void CleanUpBadPoints(){
            for (auto it = point_3d_.cbegin(); it != point_3d_.cend() /* not hoisted */; /* no increment */)
            {
            if (it->second->IsBad())
            {
                point_3d_.erase(it++);    // or "it = m.erase(it)" since C++11
            }
            else
            {
                ++it;
            }
            } 

        }


        /** @brief Adds frame to map
        @param frame_id - unique identifier for Frame in the map
        @param frame - shared pointer pointing to a created Frame object
        */

        void AddFrame(int frame_id, std::shared_ptr<Frame> frame) {
            // TODO: add warning for the duplicate
            frames_[frame_id] = frame;
        }

        /** @brief Adds point to map
        @param point_id - unique identifier for Point3D object in the map
        @param point_3d - shared pointer pointing to a created Point object
        */
        void AddPoint3D(int point_id, std::shared_ptr<Point3D> point_3d) {
            // TODO: add warning for the duplicate
            if(point_3d_.find(point_id)!=point_3d_.end()){
                throw std::invalid_argument("Duplicate point_id in AddPoint3D");
            }
            point_3d_[point_id] = point_3d;
        }

        // Adds multiple points to map with one function call, gets running indexing of points (id_point), which is increased inside the function,
        // matrix of 3D point locations (Nx3), pointers to frames (1&2) that see the point 
        /** @brief Adds multiple points to map with one function call
        @param id_point - running indexing of points as reference, gets increased inside the function
        @param point_3d - shared pointer pointing to a created Point object
        @param frame1 - shared pointer pointing to a created Frame object that sees the point
        @param uv1 - imagepoints in frame 1
        @param desc1 - descriptors in frame 1
        @param frame2 - shared pointer pointing to a created Frame object that sees the point
        @param uv2 - imagepoints in frame 2
        @param desc2 - descriptors in frame 2
        @param inlierMask - inlierMask containing 1 in the row if the corresponding rows in imagepoints and descriptors should be added
        */
        
        void AddPoints3D(int& id_point, cv::Mat points_3d, std::shared_ptr<Frame> frame1, cv::Mat uv1, cv::Mat desc1, std::shared_ptr<Frame> frame2, cv::Mat uv2, cv::Mat desc2, cv::Mat inlierMask){         
            for (int i = 0; i < points_3d.rows; i++) {
                int mask_val = inlierMask.at<uchar>(i);
                //std::cout << "Mask value: "<< mask_val << ", z location: " << (points_3d.at<double>(i,2)) << std::endl;
                if( (mask_val == 1)){
                    // make sure it is normalized, use helper segment to make euclidean
                    // cv::Mat location_3D = segment(points_3d.row(i) / points_3d.at<double>(i,3), 0, 3); 
                    cv::Mat location_3D = points_3d.row(i);
                    std::shared_ptr<Point3D> pt_object(new Point3D(id_point, location_3D));
                    pt_object->AddFrame(frame1, uv1.row(i), desc1.row(i));
                    pt_object->AddFrame(frame2, uv2.row(i), desc2.row(i));
                    (*this).AddPoint3D(id_point, pt_object);
                    id_point++;;
                }
            }
        }
 
        /** @brief Gets ids of points that are visible to frame
        @param frame_id - id of the frame
        @return vector of point ids
        */

        std::vector<int> GetPointsVisibleToFrame(int frame_id) {
            std::vector<int> point_id_list;
            for(auto it = point_3d_.begin(); it != point_3d_.end(); ++it) {
                if(it->second->IsVisibleTo(frame_id)){
                    point_id_list.push_back(it->first);
                }
            }
            return point_id_list;
        }

        /** @brief Gets ids of points that are visible to frames
        @param frame_id - ids of the frame
        @return vector of point ids
        */

        std::vector<int> GetPointsVisibleToFrames(std::vector<int> frame_id_list) {
            std::vector<int> point_id_list;
            for(auto it = point_3d_.begin(); it != point_3d_.end(); ++it) {
                for (auto it2 = frame_id_list.begin(); it2 != frame_id_list.end(); ++it2) {
                    // it contrain key values, take value which is pointer to object instance which has Is visible function
                    continue;
                    //visibility.push_back(((*it).second)->IsVisibleTo(*it2));
                }
            }
            return point_id_list;
        }

        /** @brief Gets information about the map points that are visible to frame with frame id

        @param frame_id - id of the frame
        @return tuple of imagepoints, descriptors, 3d locations, point ids of map points corresponding to the frame
        */
        std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> GetImagePointsWithFrameID(int frame_id) {
            //std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat, int>> ret; // return vector of tuples image points, desctiptors and point 3d eigen vectors
            std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<int>> ret; // return tuple image points, descriptors and point 3d vectors
            for(auto ptr_point_obj = point_3d_.begin(); ptr_point_obj != point_3d_.end(); ++ptr_point_obj) {
                if(ptr_point_obj->second->IsVisibleTo(frame_id)){
                    // Get the imagepoint and feature corresponding to the frame
                    std::tuple<cv::Mat, cv::Mat>  imgpoint_and_feature = ptr_point_obj->second->GetImagePointAndFeature(frame_id);
                    cv::Mat location_3d = ptr_point_obj->second->Get3dPoint();
                    int point_id = ptr_point_obj->second->GetID();
                    std::get<0>(ret).push_back(std::get<0>(imgpoint_and_feature)); std::get<1>(ret).push_back(std::get<1>(imgpoint_and_feature)); std::get<2>(ret).push_back(location_3d); std::get<3>(ret).push_back(point_id);                  }
            }
            return ret;

        }


        /** @brief Gets 3d locations of points with a vector of ids

        @param id_list - ids of points
        @return vector of 3d locations of points
        */
        std::vector<cv::Mat> Get3DPointsWithIDs(std::vector<int> id_list) {
            std::vector<cv::Mat> points;
            for (auto it = id_list.begin(); it != id_list.end(); ++it) {
                points.push_back(point_3d_[*it]->Get3dPoint());
            }
            return points;
        }

        /** @brief Gets all 3d locations of points in the map
        @return vector of 3d locations of points
        */

        std::vector<cv::Mat> GetAll3DPoints() {
            std::vector<cv::Mat> all_points;
            for (auto it = point_3d_.begin(); it != point_3d_.end(); ++it) {
                all_points.push_back(it->second->Get3dPoint());
            }
            return all_points;
        }

       /** @brief Gets all poses of Frame objects stored in the map
        @return vector of poses
        */
        std::vector<cv::Mat> GetAllPoses() {
            std::vector<cv::Mat> allposes;
            for(auto it = frames_.begin(); it != frames_.end(); ++it) {
                cv::Mat temp = it->second->GetPose();
                allposes.push_back(temp);
            }
            return allposes;
        }

        /** @brief Gets all translation components (xyz camera locations) of poses of Frame objects stored in the map
        @return vector of camera locations
        */
        std::vector<cv::Mat> GetAllCameraLocations(bool keyframes_only = false) {
                std::vector<cv::Mat> all_cam_locs;
                for(auto it = frames_.begin(); it != frames_.end(); ++it) {
                    if(keyframes_only && it->second->IsKeyFrame()){
                        cv::Mat temp = -GetRotation(it->second->GetPose()).inv() * GetTranslation(it->second->GetPose());
                        all_cam_locs.push_back(temp.t());
                    }else if(!keyframes_only){
                        cv::Mat temp = -GetRotation(it->second->GetPose()).inv() * GetTranslation(it->second->GetPose());
                        all_cam_locs.push_back(temp.t());
                    }
                    
                }
                return all_cam_locs;
        }
        /** @brief Updates the pose of the frame to new_pose 
        @param id id of the frame
        */
        void UpdatePose(cv::Mat new_pose, int frame_id) {
            // TODO check if frame with frame_id even exist yet
            if(frames_.find(frame_id)==frames_.end()){
                throw std::invalid_argument("No such frame_id exists in map in UpdatePose: " + frame_id);
            }
            frames_[frame_id]->UpdatePose(new_pose); // TODO: function to convert eigen pose to open cv mat and wise versa
        }

        /** @brief Updates the 3d location of the point 
        @param id id of the point
        */
        void UpdatePoint3D(cv::Mat new_point, int point_id) {
            if(point_3d_.find(point_id)==point_3d_.end()){
                throw std::invalid_argument("No such point_id exists in map in UpdatePoint3D: " + point_id);
            }
            point_3d_[point_id]->UpdatePoint(new_point);
        }

        /** @brief Gets the pointer to frame if the frame id exists in map
        @param id id of the frame
        */
        std::shared_ptr<Frame> GetFrame(int frame_id) {
            if (frames_.find(frame_id) == frames_.end()) {
                std::cout << "Trying to Get non-existent frame" << std::endl;
                return nullptr;
            } else {
                return frames_[frame_id];
            }
        }
        /** @brief Gets the pointer to point if the point id exists in map
        @param id id of the point
        */
        std::shared_ptr<Point3D> GetPoint(int point_id) {
            if (point_3d_.find(point_id) == point_3d_.end()) {
            // not found
            throw std::invalid_argument("No such point_id exists in map in GetPoint: " + point_id);
            } else {
            // found
            return point_3d_[point_id];
            }
        }

        /** @brief Stores multiple Point3D objects in the map at once
        @param points_map std::map with point ids as keys, and Point3D objects as values
        */
        void Store3DPoints(std::map<int, std::shared_ptr<Point3D>> points_map) {
            point_3d_.insert(points_map.begin(), points_map.end());
        }
        
        /** @brief compilation of calls when frame is inserted to map
        @param parent_id id of the parent frame
        @param frame_obj pointer to the Frame object itself
        @param rel_pos_trans relative pose transformation between parent frame camera pose and frame camera pose
        @param pose frame camera pose
        */
        void AddParentAndPose(int parent_id, int frame_id, std::shared_ptr<Frame> frame_obj, cv::Mat rel_pose_trans, cv::Mat pose) {
            frame_obj->AddParent(parent_id, rel_pose_trans);
            frame_obj->AddPose(pose);
            frame_obj->AddID(frame_id);
            this->AddFrame(frame_id, frame_obj);
        }

        /** @brief compilation of calls when frame is inserted to map
        @param parent_id id of the parent frame
        @param frame_obj pointer to the Frame object itself
        @param pose frame camera pose
        @param rel_pos_trans relative pose transformation between parent frame camera pose and frame camera pose
        */

        void AddPoseNode(int frame_id, std::shared_ptr<Frame> frame_obj, cv::Mat pose, int parent_id, cv::Mat rel_pose_trans) {
            frame_obj->AddParent(parent_id, rel_pose_trans);
            frame_obj->AddPose(pose);
            frame_obj->AddID(frame_id);
            this->AddFrame(frame_id, frame_obj);
        }

        /** @brief Gets all frame ids in the map
        @return vector of frame ids
        */

        std::vector<int> GetAllFrameIDs() {
            std::vector<int> frame_id_list;
            for(auto it = frames_.begin(); it != frames_.end(); ++it) {
                int frame_id = it->first;
                frame_id_list.push_back(frame_id);
            }
            return frame_id_list;

        }

        /** @brief Gets all point ids in the map
        @return vector of point ids
        */
        std::vector<int> GetAllPointIDs() {
            std::vector<int> point_id_list;
            for(auto it = point_3d_.begin(); it != point_3d_.end(); ++it) {
                int point_id = it->first;
                point_id_list.push_back(point_id);
            }
            return point_id_list;
        }
        /** @brief Adds point to frame correspondences from points to frames
        @param point_ids vector of point ids
        @param image_points image points in frame for points
        @param descriptors descriptors for image points in frame for points
        @param frame_ptr pointer to Frame object
        @param inliers vector of inlier indices to be used
        */
        void AddPointToFrameCorrespondances(std::vector<int> point_ids, cv::Mat image_points, cv::Mat descriptors, std::shared_ptr<Frame> frame_ptr, cv::Mat inliers){
            for(int i = 0; i < inliers.rows; i++){
                int inlier_idx = inliers.at<int>(i,0);
                GetPoint(point_ids[inlier_idx])->AddFrame(frame_ptr, image_points.row(inlier_idx), descriptors.row(inlier_idx));
            }
        }
        /** @brief Optimizes poses and 3D locations of points in the map, Leverberg-Marquadt used to minimize the reprojection error. Updates points and poses to the optimized values.
        @param tracking boolean flag, set true if points are set as fixed (MotionOnly)
        @param scale boolean flag, if scaling should be done for depth of points. If true, scales points so that median depth for points is 1.
        @param verbose should we print information about the optimization process
        @param n_iterations how many iterations to run the optimization algorithm
        */
        void BundleAdjustement(bool tracking, bool scale=false, bool verbose = false, int n_iterations = 10){
            double fx = 535.4; double fy = 539.2; double cx = 320.1; double cy = 247.6;
            // set up BA solver
            typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block; 
            std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<Block::PoseMatrixType>()); 
            std::unique_ptr<Block> solver_ptr( new Block(std::move(linearSolver)));
            g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) );
            g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm ( solver );
            
            std::vector<int> frame_id_list = this->GetAllFrameIDs();
            std::vector<int> point_id_list = this->GetAllPointIDs();

            // loop trough all the frames
            for(auto it = frame_id_list.begin(); it != frame_id_list.end(); ++it) {
                int frame_id = *it;
                if( this->GetFrame(frame_id)->IsKeyFrame() && !tracking){
                    cv::Mat pose_cv = this->GetFrame(frame_id)->GetPose();
                    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                    vSE3->setEstimate(toSE3Quat(pose_cv));
                    vSE3->setId((*it)*2);
                    vSE3->setFixed(*it==0);
                    optimizer.addVertex(vSE3);
                    //std::cout << "Adding to graph pose: " << frame_id << std::endl;
                }else if(tracking){
                    cv::Mat pose_cv = this->GetFrame(frame_id)->GetPose();
                    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
                    vSE3->setEstimate(toSE3Quat(pose_cv));
                    vSE3->setId((*it)*2);
                    vSE3->setFixed(*it==0);
                    optimizer.addVertex(vSE3);
                }
                
            }

            const float thHuber2D = sqrt(5.99);
            // loop through all points and (inside) create edges to frames that see the point
            for(auto it = point_id_list.begin(); it != point_id_list.end(); ++it) {
                int point_id = *it;
                g2o::VertexPointXYZ* vPoint = new g2o::VertexPointXYZ();
                vPoint->setEstimate(toVector3d((this->GetPoint(point_id)->Get3dPoint()).t()));
                vPoint->setId(point_id*2+1);
                vPoint->setMarginalized(true);
                vPoint->setFixed(tracking);
                optimizer.addVertex(vPoint);
                // GetFrames() returns std::map<int, std::tuple<std::shared_ptr<Frame>, cv::Mat, cv::Mat>> frames_; // map of frames that see this  particular point object
                auto it2_start = this->GetPoint(point_id)->GetFrames().begin();
                auto it2_end = this->GetPoint(point_id)->GetFrames().end();
                for(auto it2 = it2_start; it2 != it2_end; it2++){

                    if(optimizer.vertex((it2->first)*2) == NULL){
                        continue;
                    }

                    Eigen::MatrixXd uv;
                    cv::cv2eigen(std::get<1>(it2->second).t(), uv);
                    ////std::cout << "Image projection on frame " << (it2->first)*2 << ": " << uv << std::endl;
                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(point_id*2+1)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((it2->first)*2)));
                    e->setMeasurement(uv);
                    e->setId(point_id+100000);
                    e->setInformation(Eigen::Matrix2d::Identity());
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                    e->fx = fx;
                    e->fy = fy;
                    e->cx = cx;
                    e->cy = cy;
                    optimizer.addEdge(e);
                }
            }
            //optimizer.save("beforeopt.g2o");
            optimizer.initializeOptimization();
            optimizer.setVerbose(verbose);
            optimizer.optimize(n_iterations);
            //optimizer.save("afteropt.g2o");

            double median_depth = 1;
            if(scale){
                int num_points = 0;
                std::vector<double> median_vec;
                for(auto it = point_id_list.begin(); it != point_id_list.end(); ++it) {
                    int point_id = *it;
                    g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_id*2+1));
                    ////std::cout << "Updating to 3D location: " << toCvMat(vPoint->estimate()) << std::endl;
                    median_vec.push_back( cv::norm(toCvMat(vPoint->estimate()).t()) );
                }
                std::sort(median_vec.begin(), median_vec.end()); // sort so that median depth is middle element
                median_depth = median_vec[median_vec.size()/2];
            }
            for(auto it = frame_id_list.begin(); it != frame_id_list.end(); ++it) {
                int frame_id = *it;
                if(optimizer.vertex(frame_id*2) == NULL){
                    continue;
                }
                g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id*2));
                g2o::SE3Quat SE3quat = vSE3->estimate();
                ////std::cout << "BundleAdjustement updates pose " << frame_id << " to: " << toCvMat(SE3quat) << std::endl;
                GetFrame(frame_id)->UpdatePose(NormalizeTranslation(toCvMat(SE3quat), median_depth));
            }
            for(auto it = point_id_list.begin(); it != point_id_list.end(); ++it) {
                int point_id = *it;
                g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_id*2+1));
                ////std::cout << "Updating to 3D location: " << toCvMat(vPoint->estimate()) << std::endl;
                GetPoint(point_id)->UpdatePoint(toCvMat(vPoint->estimate()).t()/median_depth);
            }    
        }

        

    private:
        std::map<int, std::shared_ptr<Frame>> frames_; //!<frame map container hold unique frame id's as a key and std::shared_ptr<Frame> as a value
        std::map<int, std::shared_ptr<Point3D>> point_3d_; //!< point_3d_ map container hold unique point id's as a key and std::shared_ptr<Point3D> as a value
};

#endif
