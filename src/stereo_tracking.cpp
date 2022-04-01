#include "constants.h"
#include "types.hpp"

#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>

using namespace vslam;

class StereoTracking {

    ros::Subscriber imu_sub;
    std::deque<sensor_msgs::Imu> imu_queue;

    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_crosscheck_;

    int num_inliers_ = 0; // number of inliers after RANSAC

    SE3 T_c_l_ = SE3(); // T_current(camera)_last(camera)
    SE3 T_c_w_ = SE3(); // T_current(camera)_world

    int seq_ = 1; // sequence number

    Frame current_frame;
    Frame last_frame;
    int success;



public:
    StereoTracking():
    it_(nh),
    orig_image_sub_( it_, camera_topic_1, 1 ),
    warp_image_sub_( it_, camera_topic_2, 1 ),
    sync( MySyncPolicy( 10 ), orig_image_sub_, warp_image_sub_ )
    {
     
       sync.registerCallback( boost::bind( &StereoTracking::imageCallback, this, _1, _2 ) );
       detector_ = cv::ORB::create(3000);
       descriptor_ = cv::ORB::create();
       matcher_crosscheck_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    }


   
    void imageCallback(
        const sensor_msgs::ImageConstPtr& msg,
        const sensor_msgs::ImageConstPtr& warp_msg
    );

    int feature_detection(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);    
    void adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints,
                                         const int num);

    int disparity_map(const Frame &frame, cv::Mat &disparity);


 private:   
    ros::NodeHandle nh;
    image_transport::ImageTransport it_;
    typedef image_transport::SubscriberFilter ImageSubscriber;  
    ImageSubscriber orig_image_sub_;
    ImageSubscriber warp_image_sub_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image
    > MySyncPolicy;

    message_filters::Synchronizer< MySyncPolicy > sync;  

};

  void StereoTracking::imageCallback(
    const sensor_msgs::ImageConstPtr& left_img_msg,
    const sensor_msgs::ImageConstPtr& right_img_msg
  ){

      current_frame = Frame();
      current_frame.left_img_ = cv_bridge::toCvShare(left_img_msg , "bgr8")->image;
      current_frame.right_img_ = cv_bridge::toCvShare(right_img_msg , "bgr8")->image;
      
      success = feature_detection(current_frame.left_img_, current_frame.keypoints, current_frame.desc);
      success = disparity_map(current_frame, current_frame.disparity_);  
    
 }


int StereoTracking::feature_detection(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // ensure the image is read
    if (!img.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // feature detection (Oriented FAST)
    detector_->detect(img, keypoints);

    adaptive_non_maximal_suppresion(keypoints, 500);

    // BRIEF describer
    descriptor_->compute(img, keypoints, descriptors);

    // show output image
    //cv::Mat outimg1;
    //cv::drawKeypoints(img, keypoints, outimg1);
    //cv::imshow("ORB features", outimg1);
    //cv::waitKey(1);

    return 0;
}

void StereoTracking::adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints,
                                         const int num)
{
    // if number of keypoints is already lower than the threshold, return
    if (keypoints.size() < num)
    {
        return;
    }

    // sort the keypoints according to its reponse (strength)
    std::sort(keypoints.begin(), keypoints.end(), [&](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {
        return lhs.response > rhs.response;
    });

    // vector for store ANMS points
    std::vector<cv::KeyPoint> ANMSpt;

    std::vector<double> rad_i;
    rad_i.resize(keypoints.size());

    std::vector<double> rad_i_sorted;
    rad_i_sorted.resize(keypoints.size());

    // robust coefficient: 1/0.9 = 1.1
    const float c_robust = 1.11;

    // computing the suppression radius for each feature (strongest overall has radius of infinity)
    // the smallest distance to another point that is significantly stronger (based on a robustness parameter)
    for (int i = 0; i < keypoints.size(); ++i)
    {
        const float response = keypoints.at(i).response * c_robust;

        // maximum finit number of double
        double radius = std::numeric_limits<double>::max();

        for (int j = 0; j < i && keypoints.at(j).response > response; ++j)
        {
            radius = std::min(radius, cv::norm(keypoints.at(i).pt - keypoints.at(j).pt));
        }

        rad_i.at(i) = radius;
        rad_i_sorted.at(i) = radius;
    }

    // sort it
    std::sort(rad_i_sorted.begin(), rad_i_sorted.end(), [&](const double &lhs, const double &rhs) {
        return lhs > rhs;
    });

    // find the final radius
    const double final_radius = rad_i_sorted.at(num - 1);
    for (int i = 0; i < rad_i.size(); ++i)
    {
        if (rad_i.at(i) >= final_radius)
        {
            ANMSpt.push_back(keypoints.at(i));
        }
    }

    // swap address to keypoints, O(1) time
    keypoints.swap(ANMSpt);
}


int StereoTracking::disparity_map(const Frame &frame, cv::Mat &disparity)
{
     cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
      0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);

    cv::Mat disparity_sgbm;
    sgbm->compute(frame.left_img_, frame.right_img_, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(1);

    return 0;
}





int main(int argc, char **argv) {
    ros::init(argc, argv, "slam_gtsam");

    StereoTracking lt;
    
    ros::spin();
    return 0;

}






