#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <pcl/register_point_struct.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <nav_msgs/Odometry.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


#include <string>

// Newer College Dataset

const std::string imu_topic = "/os1_cloud_node/imu";
const std::string camera_topic_1 = "/camera/infra1/image_rect_raw";
const std::string camera_topic_2 = "/camera/infra2/image_rect_raw";



const float f_x = 431.3873911369959;
const float f_y =  430.2496176152663;
const float u = 427.4407802012019;
const float v = 238.52694867508183;  

float k_data[9] = {f_x, 0, u, 0, f_y, v, 0, 0, 1};
cv::Mat K(3, 3, CV_32FC1, k_data);

float dist_data[4] = { 431.3873911369959 ,  430.2496176152663, 427.4407802012019, 238.52694867508183};
cv::Mat distrotion(1, 4, CV_32FC1, dist_data);


#endif
