#include "constants.h"
#include "types.hpp"
#include "map.hpp"

#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>




#include <vector>

using namespace std;


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

    Frame frame_current_;
    Frame last_frame;
    int success;
    bool initializion = false; 
    std::queue<int> landmark_queue;
    bool if_insert_keyframe;
    bool check;
    Map &my_map_;



    ///Optimization



    // Create an iSAM2 object. Unlike iSAM1, which performs periodic batch steps
    // to maintain proper linearization and efficient variable ordering, iSAM2
    // performs partial relinearization/reordering at each step. A parameter
    // structure is available that allows the user to set various properties, such
    // as the relinearization threshold and type of linear solver. For this
    // example, we we set the relinearization threshold small so the iSAM2 result
    // will approach the batch result.
    //ISAM2Params parameters;

    //NonlinearFactorGraph graph;
    //Values initial;

public:
    StereoTracking(ros::NodeHandle &nh, Map &map):
    it_(nh),
    orig_image_sub_( it_, camera_topic_1, 1 ),
    warp_image_sub_( it_, camera_topic_2, 1 ),
    my_map_(map),
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
    std::vector<bool> set_ref_3d_position(std::vector<cv::Point3f> &pts_3d,
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::Mat &descriptors,
                                          Frame &frame);

    int feature_matching(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2, std::vector<cv::DMatch> &feature_matches,  Frame &frame);
    int intialize(Frame &frame,  std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, std::vector<cv::Point3f> &pts_3d);
    void motion_estimation(Frame &frame);
    bool check_motion_estimation(Frame frame_current_);
    bool insert_key_frame(bool check, std::vector<cv::Point3f> &pts_3d, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    /*
    // Optimization functions
    void add_keyframe_gtsam( int keyframe_id );
    void initialize_graph_gtsam();
    void update_gtsam();
    int add_landmarks_gtsam();
    */

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

    int curr_landmark_id_ = 0;
    int image_seq_ = 0;
    int keyframe_id_ = 0;




};

  void StereoTracking::imageCallback(
    const sensor_msgs::ImageConstPtr& left_img_msg,
    const sensor_msgs::ImageConstPtr& right_img_msg
  ){

      std::vector<cv::Point3f> pts_3d;
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      std::vector<cv::DMatch> feature_matches;

      Frame frame = Frame();
      frame.left_img_ = cv_bridge::toCvShare(left_img_msg , "bgr8")->image;
      frame.right_img_ = cv_bridge::toCvShare(right_img_msg , "bgr8")->image;
      success = feature_detection(frame.left_img_, keypoints, descriptors);
      success = disparity_map(frame, frame.disparity_); 
      std::cout << "keyframe: " << keyframe_id_ << std::endl;

      if(!initializion)
      {
            // find out if the keypoint are legit
            success = intialize(frame, keypoints, descriptors, pts_3d);
            last_frame = frame;
            initializion = true;


      }
    else
      {


        if (last_frame.is_keyframe_ == true)
        {
            last_frame = my_map_.keyframes_.at(last_frame.keyframe_id_);
            
        }

        frame.frame_id_ = image_seq_;
        std::cout << "last_frame: " << last_frame.frame_id_ << std::endl;
        std::cout << "current_frame: " << image_seq_ << std::endl;

        cv::Mat descriptors_last;
        std::vector<cv::KeyPoint> keypoints_last;

        std::cout << "last_frame features size : " << last_frame.features_.size() << std::endl;

        for (int i = 0; i < last_frame.features_.size(); i++)
        {
            descriptors_last.push_back(last_frame.features_.at(i).descriptor_);
            keypoints_last.push_back(last_frame.features_.at(i).keypoint_);
        }

        success = feature_matching(descriptors_last, descriptors, feature_matches, frame);

        std::cout << "feature match size : " <<  feature_matches.size() << std::endl;
        
        if( feature_matches.size() > 4) 
        {
                for( int i = 0; i < feature_matches.size(); i++ )
                {

                    Feature feature_to_add(i, image_seq_, keypoints.at( feature_matches.at(i).trainIdx )
                                                        , descriptors.row( feature_matches.at(i).trainIdx ));
                    // build the connection from feature to landmark
                    

                    feature_to_add.landmark_id_ = last_frame.features_.at( feature_matches.at(i).queryIdx).landmark_id_;
                    frame.features_.push_back(feature_to_add);

                }

                
                motion_estimation(frame);

                frame.T_c_w_ = T_c_w_;

                T_c_l_ = frame.T_c_w_ * last_frame.T_c_w_.inverse();


                check = check_motion_estimation(frame);
                //check = true;
                frame_current_ = frame;
                if_insert_keyframe = insert_key_frame(check, pts_3d, keypoints, descriptors);
                last_frame = frame_current_;
        }

      }     
      
    image_seq_++;

   
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
    cv::Mat outimg1;
    cv::drawKeypoints(img, keypoints, outimg1);
    cv::imshow("ORB features", outimg1);
    cv::waitKey(5);

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

    //cv::imshow("disparity", disparity / 96.0);
    //cv::waitKey(1);

    return 0;
}



std::vector<bool> StereoTracking::set_ref_3d_position(std::vector<cv::Point3f> &pts_3d,
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::Mat &descriptors,
                                          Frame &frame)
{
    // clear existing 3D positions
    pts_3d.clear();

    // create filetered descriptors for replacement
    cv::Mat descriptors_last_filtered;
    std::vector<cv::KeyPoint> keypoints_last_filtered;
    std::vector<bool> reliable_depth;
    std::cout << "keypoints size" << keypoints.size() << std::endl;
    
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        Eigen::Vector3d relative_pos_3d;
        Eigen::Vector3d pos_3d = frame.find_3d(keypoints.at(i), relative_pos_3d);
        // filer out points with no depth information (disparity value = -1)
        if (relative_pos_3d(2) > 1 && relative_pos_3d(2) < 40)
        {
            pts_3d.push_back(cv::Point3f(pos_3d(0), pos_3d(1), pos_3d(2)));
            descriptors_last_filtered.push_back(descriptors.row(i));
            keypoints_last_filtered.push_back(keypoints.at(i));

            // mark reliable depth information
            if (relative_pos_3d(2) < 40)
            {
                reliable_depth.push_back(true);
            }
            else
            {
                reliable_depth.push_back(false);
            }
        }
    }

    // copy the filtered descriptors;
    descriptors = descriptors_last_filtered;
    keypoints = keypoints_last_filtered;

    return reliable_depth;
}

int StereoTracking::feature_matching(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2, std::vector<cv::DMatch> &feature_matches, Frame &frame)
{
    feature_matches.clear();

    std::vector<cv::DMatch> matches_crosscheck;
    // use cross check for matching
    matcher_crosscheck_->match(descriptors_1, descriptors_2, matches_crosscheck);
    // std::cout << "Number of matches after cross check: " << matches_crosscheck.size() << std::endl;

    // calculate the min/max distance
    auto min_max = minmax_element(matches_crosscheck.begin(), matches_crosscheck.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.distance < rhs.distance;
    });

    auto min_element = min_max.first;
    auto max_element = min_max.second;
    // std::cout << "Min distance: " << min_element->distance << std::endl;
    // std::cout << "Max distance: " << max_element->distance << std::endl;

    // threshold: distance should be smaller than two times of min distance or a give threshold
    double frame_gap = frame.frame_id_ - last_frame.frame_id_;
    for (int i = 0; i < matches_crosscheck.size(); i++)
    {
        if (matches_crosscheck.at(i).distance <= std::max(2.0 * min_element->distance, 30.0 * frame_gap))
        {
            feature_matches.push_back(matches_crosscheck.at(i));
        }
    }

    // std::cout << "Number of matches after threshold: " << feature_matches.size() << std::endl;

    return 0;
}

void StereoTracking::motion_estimation(Frame &frame)
{
    // 3D positions from the last frame
    // 2D pixels in current frame
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (int i = 0; i < frame.features_.size(); i++)
    {
        int landmark_id = frame.features_.at(i).landmark_id_;
        if (landmark_id == -1)
        {
            std::cout << "No landmark associated!" << std::endl;
        }

        pts3d.push_back(my_map_.landmarks_.at(landmark_id).pt_3d_);
        pts2d.push_back(frame.features_.at(i).keypoint_.pt);
    }



    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);

    num_inliers_ = inliers.rows;
    std::cout << "Number of PnP inliers: " << num_inliers_ << std::endl;

    // transfer rvec to matrix
    cv::Mat SO3_R_cv;
    cv::Rodrigues(rvec, SO3_R_cv);
    Eigen::Matrix3d SO3_R;
    SO3_R << SO3_R_cv.at<double>(0, 0), SO3_R_cv.at<double>(0, 1), SO3_R_cv.at<double>(0, 2),
        SO3_R_cv.at<double>(1, 0), SO3_R_cv.at<double>(1, 1), SO3_R_cv.at<double>(1, 2),
        SO3_R_cv.at<double>(2, 0), SO3_R_cv.at<double>(2, 1), SO3_R_cv.at<double>(2, 2);

    T_c_w_ = SE3(
        SO3_R,
        Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

    // mark inlieres
    for (int idx = 0; idx < inliers.rows; idx++)
    {
        int index = inliers.at<int>(idx, 0);
        frame.features_.at(index).is_inlier = true;

        // also mark the landmark
        // currently all landmarks are initialized to true
        // my_map_.landmarks_.at(frame.features_.at(index).landmark_id_).is_inlier = true;
    }

    // delete outliers
    frame.features_.erase(std::remove_if(
                              frame.features_.begin(), frame.features_.end(),
                              [](const Feature &x) {
                                  return x.is_inlier == false;
                              }),
                          frame.features_.end());

    // std::cout << "T_c_l Translation x: " << tvec.at<double>(0, 0) << "; y: " << tvec.at<double>(1, 0) << "; z: " << tvec.at<double>(2, 0) << std::endl;
}



bool StereoTracking::check_motion_estimation(Frame frame_current_)
{
    // check the number of inliers
    if (num_inliers_ < 10)
    {
        std::cout << "Frame id: " << last_frame.frame_id_ << " and " << frame_current_.frame_id_ << std::endl;
        std::cout << "Rejected - inliers not enough: " << num_inliers_ << std::endl;
        return false;
    }

    // check if the motion is too large
    Sophus::Vector6d displacement = T_c_l_.log();
    double frame_gap = frame_current_.frame_id_ - last_frame.frame_id_; // get the idx gap between last and current frame
    std::cout << "displacement: " << displacement.norm() << std::endl;
    if (displacement.norm() > (10.0 * frame_gap))
    {
        std::cout << "Frame id: " << last_frame.frame_id_ << " and " << frame_current_.frame_id_ << std::endl;
        std::cout << "Rejected - motion is too large: " << displacement.norm() << std::endl;
        return false;
    }

    // check if the motion is forward
    // Eigen::Vector3d translation = T_c_l_.translation();
    // if (translation(2) > 1)
    // {
    //     std::cout << "Frame id: " << frame_last_.frame_id_ << " and " << frame_current_.frame_id_ << std::endl;
    //     std::cout << "Rejected - motion is backward: " << translation(2) << std::endl;
    //     return false;
    // }

    return true;
}

bool StereoTracking::insert_key_frame(bool check, std::vector<cv::Point3f> &pts_3d, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    // if the number of inliers is enough or the frame is rejected
    // parameter tunning
    // added more keyframes when turning
    if ((num_inliers_ >= 80 && T_c_l_.angleY() < 0.03) || check == false)
    {
        return false;
    }

    // fill the extra information
    frame_current_.is_keyframe_ = true;
    frame_current_.keyframe_id_ = keyframe_id_;

    // add observations
    for (int i = 0; i < frame_current_.features_.size(); i++)
    {
        int landmark_id = frame_current_.features_.at(i).landmark_id_;
        my_map_.landmarks_.at(landmark_id).observed_times_++;
        Observation observation(frame_current_.keyframe_id_, frame_current_.features_.at(i).feature_id_);
        my_map_.landmarks_.at(landmark_id).observations_.push_back(observation);
        my_map_.landmarks_.at(landmark_id).observations_queue_.push(observation);
        landmark_queue.push( landmark_id);

        // std::cout << "Landmark " << landmark_id << " "
        //           << "has obsevation times: " << my_map_.landmarks_.at(landmark_id).observed_times_ << std::endl;
        // std::cout << "Landmark " << landmark_id << " "
        //           << "last observation keyframe: " << my_map_.landmarks_.at(landmark_id).observations_.back().keyframe_id_ << std::endl;
    }

    // add more features with triangulated points to the map
    disparity_map(frame_current_, frame_current_.disparity_);
    std::vector<bool> reliable_depth = set_ref_3d_position(pts_3d, keypoints, descriptors, frame_current_);

    // calculate the world coordinate
    // no relative motion any more

    int feature_id = frame_current_.features_.size();
    // if the feature does not exist in the frame already, add it
    for (int i = 0; i < keypoints.size(); i++)
    {
        bool exist = false;
        for (auto &feat : frame_current_.features_)
        {
            if (feat.keypoint_.pt.x == keypoints.at(i).pt.x && feat.keypoint_.pt.y == keypoints.at(i).pt.y)
            {
                exist = true;

                // try to update the landmark position if already exist
                if ((my_map_.landmarks_.at(feat.landmark_id_).reliable_depth_ == false) && (reliable_depth.at(i) == true))
                {
                    my_map_.landmarks_.at(feat.landmark_id_).pt_3d_ = pts_3d.at(i);
                    my_map_.landmarks_.at(feat.landmark_id_).reliable_depth_ = true;
                }
            }
        }
        if (exist == false)
        {
            // add this feature
            // put the features into the frame with feature_id, frame_id, keypoint, descriptor
            // build the connection from feature to frame
            Feature feature_to_add(feature_id, frame_current_.frame_id_,
                                   keypoints.at(i), descriptors.row(i));

            // build the connection from feature to landmark
            feature_to_add.landmark_id_ = curr_landmark_id_;
            frame_current_.features_.push_back(feature_to_add);
            // create a landmark
            // build the connection from landmark to feature
            Observation observation(frame_current_.keyframe_id_, feature_id);
            Landmark landmark_to_add(curr_landmark_id_, pts_3d.at(i), descriptors.row(i), reliable_depth.at(i), observation);
            curr_landmark_id_++;
            // insert the landmark
            my_map_.insert_landmark(landmark_to_add);
            feature_id++;
        }
    }
    keyframe_id_++;

    // insert the keyframe
    my_map_.insert_keyframe(frame_current_);

    // std::cout << "Set frame: " << frame_current_.frame_id_ << " as keyframe "
    //           << frame_current_.keyframe_id_ << std::endl;

    return true;
}




int StereoTracking::intialize(Frame &frame,  std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, std::vector<cv::Point3f> &pts_3d)
{
            std::vector<bool> reliable_depth;
            reliable_depth = set_ref_3d_position( pts_3d,
                                keypoints,
                                descriptors,
                                frame);

            for( int i = 0; i < keypoints.size(); i++ )
            {

                Feature feature_to_add(i, 0, keypoints.at(i), descriptors.row(i));
                // build the connection from feature to landmark
                feature_to_add.landmark_id_ = curr_landmark_id_;
                frame.features_.push_back(feature_to_add);
                // create a landmark
                // build the connection from landmark to feature
                // this 0 is also the keyframe id
                Observation observation(0, i);
                Landmark landmark_to_add(curr_landmark_id_, pts_3d.at(i), descriptors.row(i), reliable_depth.at(i), observation);
                my_map_.insert_landmark(landmark_to_add);
                curr_landmark_id_++;


            }

            frame.fill_frame(SE3(), true, keyframe_id_);
            
            keyframe_id_++; 
            my_map_.insert_keyframe(frame);
            return 1; 

   
}

/*

int StereoTracking::add_landmarks_gtsam( )
{

    // Define the camera calibration parameters
    Cal3_S2 K_gtsam(f_x, f_y, 0, u, v);

    // Define the camera observation noise model, 1 pixel stddev
    noiseModel::Isotropic::shared_ptr measurementNoise = noiseModel::Isotropic::Sigma(2, 1.0);
    int landmark_to_add;
    for(int i =0; i< landmark_queue.size(); i++)
    {
        landmark_to_add = landmark_queue.front();
        landmark_queue.pop();
        if( my_map_.landmarks_.at(landmark_to_add).added_to_graph == false )
        {   
            cv::Point3f p = my_map_.landmarks_.at(landmark_to_add).pt_3d_;
            initial.insert<Point3>(Symbol('l', landmark_to_add), Point3(p.x, p.y, p.z));
            my_map_.landmarks_.at(landmark_to_add).added_to_graph = true;
        
        }

        std::queue<Observation> observation_to_add = my_map_.landmarks_.at(landmark_to_add).observations_queue_;
        Frame frame;
        for(int i =0; i< observation_to_add.size(); i++)
        {
            Observation cur_observation = observation_to_add.front();
            observation_to_add.pop();

            frame = my_map_.keyframes_.at(cur_observation.keyframe_id_); 
            Feature add_feature_graph = frame.features_.at(cur_observation.feature_id_);

            Point2 pt;

            pt(0) = add_feature_graph.keypoint_.pt.x;
            pt(1) = add_feature_graph.keypoint_.pt.y;
            graph.emplace_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                pt, measurementNoise, Symbol('x', cur_observation.keyframe_id_), Symbol('l', landmark_to_add), K_gtsam);


        }
    }

}


void StereoTracking::add_keyframe_gtsam( int keyframe_id )
{
    Frame frame;
    frame = my_map_.keyframes_.at(keyframe_id); 

    
    Rot3 R = frame.T_c_w_.rotationMatrix;
    Point3 t = frame.T_c_w_.translation();

    Pose3 pose(R, t);
    initial.insert(Symbol('x', keyframe_id), pose );


}


void StereoTracking::initialize_graph_gtsam()
{   


      // Intentionally initialize the variables off from the ground truth
    Rot3 R = SE3().rotationMatrix;
    Point3 t = SE3().translation();

    Pose3 pose(R, t);
    initial.insert(Symbol('x', 0), pose );

    static auto kPosePrior = noiseModel::Diagonal::Sigmas(
          (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
              .finished());
    graph.addPrior(Symbol('x', 0), pose, kPosePrior);

}


void StereoTracking::update_gtsam()
{
      parameters.relinearizeThreshold = 0.01;
      parameters.relinearizeSkip = 1;
      ISAM2 isam(parameters);
      isam.update(graph, initialEstimate);
      // Each call to iSAM2 update(*) performs one iteration of the iterative
      // nonlinear solver. If accuracy is desired at the expense of time,
      // update(*) can be called additional times to perform multiple optimizer
      // iterations every step.
      isam.update();
      Values currentEstimate = isam.calculateEstimate();
      


      // Clear the factor graph and values for the next iteration
      graph.resize(0);
      graph = gtsam::NonlinearFactorGraph();
      initial.clear();

}
*/

int main(int argc, char **argv) {
    ros::init(argc, argv, "slam_gtsam");
    ros::NodeHandle nh;
    Map my_map_(nh); 

    StereoTracking lt(nh, my_map_);
    
    ros::spin();
    return 0;

}






