#ifndef TYPES_INCLUDE_GUARD_HPP
#define TYPES_INCLUDE_GUARD_HPP


#include <cmath>
#include <iostream>
#include <include_library.hpp>


namespace vslam
{

struct Frame;
struct Landmark;
struct Feature;

struct Landmark {
	cv::Point3d pt;
    int seen;
};

struct Frame
    {  
        
        int frame_id_;
        cv::Mat left_img_, right_img_;
        cv::Mat disparity_;
        SE3 T_c_w_ = SE3();
        
        cv::Mat desc; // feature descriptor
        std::vector<cv::KeyPoint> keypoints; // keypoint

        // alias to clarify map usage below
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // seypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
   };

} // namespace vslam

#endif





