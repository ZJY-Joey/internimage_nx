// Copyright (c) 2008, Willow Garage, Inc.
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of the Willow Garage nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef DEPTH_IMAGE_PROC__POINT_CLOUD_XYZRGB_HPP_
#define DEPTH_IMAGE_PROC__POINT_CLOUD_XYZRGB_HPP_

#include <memory>
#include <mutex>
#include <unordered_set>

#include <depth_image_proc/visibility.h>
#include <image_geometry/pinhole_camera_model.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/synchronizer.hpp>
#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

// PCL Headers
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/PCLPointCloud2.h> // Now explicitly used for internal storage
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h> // Essential for ROS <-> PCL conversions

// TF2 Headers
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/buffer_core.h>
#include <message_filters/cache.h>

namespace depth_image_proc
{

class PointCloudXyzrgbLabelNode : public rclcpp::Node
{
public:
  DEPTH_IMAGE_PROC_PUBLIC PointCloudXyzrgbLabelNode(const rclcpp::NodeOptions & options);

private:
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using PCLPointCloud2 = pcl::PCLPointCloud2; // Alias for PCL type

  // Subscriptions
  // Image subscriptions (depth, combined segmentation/color, confidence)
  image_transport::SubscriberFilter sub_depth_, sub_combined_;
  message_filters::Subscriber<CameraInfo> sub_info_;
  using SyncPolicy =
    message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;
  using ExactSyncPolicy =
    message_filters::sync_policies::ExactTime<Image, Image, CameraInfo>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
  using ExactSynchronizer = message_filters::Synchronizer<ExactSyncPolicy>;
  using PointCloudCache = message_filters::Cache<PointCloud2>;
  using PointCloud2ConstPtr = PointCloud2::ConstSharedPtr;
  using AddFunction = void (message_filters::Cache<PointCloud2>::* ) (PointCloud2ConstPtr);
  using PointCloud2SharedPtr = sensor_msgs::msg::PointCloud2::SharedPtr;
  using PointCloud2ConstSharedPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;



  std::shared_ptr<Synchronizer> sync_;
  std::shared_ptr<ExactSynchronizer> exact_sync_;

  // Publications
  std::mutex connect_mutex_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_point_cloud_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_ground_point_cloud_;

  image_geometry::PinholeCameraModel model_;
  PointCloud2::SharedPtr latest_lidar_pointcloud_;
  PointCloud2::SharedPtr latest_transformed_pointcloud_;
  rclcpp::Time last_processed_lidar_cloud_stamp_;

  void connectCb();

  void imageCb(
    const Image::ConstSharedPtr & depth_msg,
    const Image::ConstSharedPtr & combined_msg,
    const CameraInfo::ConstSharedPtr & info_msg);
  
  void pointcloud_callback(const PointCloud2::SharedPtr msg);
  // Filtering configuration
  std::unordered_set<uint8_t> filter_labels_{15};   // Labels to drop (default) or keep (if filter_keep_ = true)
  bool filter_keep_{false};                       // false: drop labels in set; true: keep only labels in set

  // TF2 members for pointcloud frame transformation
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::string target_frame_;
  int outlier_reject_MeanK_;
  double outlier_reject_StddevMulThresh_;

  //lidar cache
  std::shared_ptr<PointCloudCache> lidar_cache_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_pointcloud_cache_;
  static const int LIDAR_CACHE_SIZE = 100;
};

}  // namespace depth_image_proc

#endif  // DEPTH_IMAGE_PROC__POINT_CLOUD_XYZRGB_LABEL_HPP_
