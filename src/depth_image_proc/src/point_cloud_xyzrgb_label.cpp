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

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <limits>

#include "cv_bridge/cv_bridge.hpp"

#include <depth_image_proc/conversions.hpp>
#include <depth_image_proc/point_cloud_xyzrgb_label.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace depth_image_proc
{


PointCloudXyzrgbLabelNode::PointCloudXyzrgbLabelNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PointCloudXyzrgbLabelNode", options)
{
  // Read parameters
  int queue_size = this->declare_parameter<int>("queue_size", 20);
  bool use_exact_sync = this->declare_parameter<bool>("exact_sync", true);

  // Label filtering parameters
  // filter_labels: list of integer labels. By default, points with these labels will be dropped (masked as NaN).
  // Set filter_keep=true to invert behavior and keep only points with these labels.
  auto filter_labels_param = this->declare_parameter<std::vector<int64_t>>("filter_labels", std::vector<int64_t>{});
  filter_labels_param = this->get_parameter("filter_labels").as_integer_array();
  filter_keep_ = this->declare_parameter<bool>("filter_keep", false);
  filter_keep_ = this->get_parameter("filter_keep").as_bool();
  filter_labels_.clear();
  for (const auto & v : filter_labels_param) {
    if (v >= 0 && v <= 255) {
      filter_labels_.insert(static_cast<uint8_t>(v));
    }
  }

  RCLCPP_INFO(
    get_logger(), "PointCloudXyzrgbLabelNode::PointCloudXyzrgbLabelNode called");

  // Synchronize inputs. Topic subscriptions happen on demand in the connection callback.
  if (use_exact_sync) {
    exact_sync_ = std::make_shared<ExactSynchronizer>(
      ExactSyncPolicy(queue_size),
      sub_depth_,
      sub_rgb_,
      sub_id_,
      sub_info_);
    exact_sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
  } else {
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(queue_size), sub_depth_, sub_rgb_, sub_id_, sub_info_);
    sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
  }

  // Monitor whether anyone is subscribed to the output
  // TODO(ros2) Implement when SubscriberStatusCallback is available
  // ros::SubscriberStatusCallback connect_cb = boost::bind(&PointCloudXyzrgbLabelNode::connectCb, this);
  connectCb();
  // TODO(ros2) Implement when SubscriberStatusCallback is available
  // Make sure we don't enter connectCb() between advertising and assigning to pub_point_cloud_
  std::lock_guard<std::mutex> lock(connect_mutex_);
  // TODO(ros2) Implement connect_cb when SubscriberStatusCallback is available
  // pub_point_cloud_ = depth_nh.advertise<PointCloud>("points", 1, connect_cb, connect_cb);
  pub_point_cloud_ = create_publisher<PointCloud2>("points", rclcpp::SensorDataQoS());
  // TODO(ros2) Implement connect_cb when SubscriberStatusCallback is available
}

// Handles (un)subscribing when clients (un)subscribe
void PointCloudXyzrgbLabelNode::connectCb()
{
  // RCLCPP_INFO(
  //   get_logger(), "PointCloudXyzrgbLabelNode::connectCb called");
  std::lock_guard<std::mutex> lock(connect_mutex_);
  // TODO(ros2) Implement getNumSubscribers when rcl/rmw support it
  // if (pub_point_cloud_->getNumSubscribers() == 0)
  if (0) {
    // TODO(ros2) Implement getNumSubscribers when rcl/rmw support it
    sub_depth_.unsubscribe();
    sub_rgb_.unsubscribe();
    sub_id_.unsubscribe();
    sub_info_.unsubscribe();
  } else if (!sub_depth_.getSubscriber()) {
    // parameter for depth_image_transport hint
    std::string depth_image_transport_param = "depth_image_transport";
    image_transport::TransportHints depth_hints(this, "raw", depth_image_transport_param);

    rclcpp::SubscriptionOptions sub_opts;
    // Update the subscription options to allow reconfigurable qos settings.
    sub_opts.qos_overriding_options = rclcpp::QosOverridingOptions {
      {
        // Here all policies that are desired to be reconfigurable are listed.
        rclcpp::QosPolicyKind::Depth,
        rclcpp::QosPolicyKind::Durability,
        rclcpp::QosPolicyKind::History,
        rclcpp::QosPolicyKind::Reliability,
      }};

    // depth image can use different transport.(e.g. compressedDepth)
    sub_depth_.subscribe(
      this, "depth_registered/image_rect",
      depth_hints.getTransport(), rmw_qos_profile_default, sub_opts);

    // rgb uses color ros transport hints.
    image_transport::TransportHints color_hints(this, "raw");
    sub_rgb_.subscribe(
      this, "rgb/image_rect_color",
      color_hints.getTransport(), rmw_qos_profile_default, sub_opts);
    sub_info_.subscribe(this, "rgb/camera_info");

    //id uses label ros transport hints.
    image_transport::TransportHints id_hints(this, "raw");
    sub_id_.subscribe(
        this, "id/image_rect_id",
        id_hints.getTransport(), rmw_qos_profile_default, sub_opts);

  }
}

void PointCloudXyzrgbLabelNode::imageCb(
  const Image::ConstSharedPtr & depth_msg,
  const Image::ConstSharedPtr & rgb_msg_in,
  const Image::ConstSharedPtr & id_msg_in,
  const CameraInfo::ConstSharedPtr & info_msg)
{
    // RCLCPP_INFO(
    //   get_logger(), "PointCloudXyzrgbLabelNode::imageCb called");
  // Check for bad inputs of color image
  if (depth_msg->header.frame_id != rgb_msg_in->header.frame_id) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      10000,  // 10 seconds
      "Depth image frame id [%s] doesn't match RGB image frame id [%s]",
      depth_msg->header.frame_id.c_str(), rgb_msg_in->header.frame_id.c_str());
  }

  // check for bad inputs of id image
  if (depth_msg->header.frame_id != id_msg_in->header.frame_id) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      10000,  // 10 seconds
      "Depth image frame id [%s] doesn't match ID image frame id [%s]",
      depth_msg->header.frame_id.c_str(), id_msg_in->header.frame_id.c_str());
  }

  // Update camera model
  model_.fromCameraInfo(info_msg);

  // Check if the input color image has to be resized
  Image::ConstSharedPtr rgb_msg = rgb_msg_in;
  if (depth_msg->width != rgb_msg->width || depth_msg->height != rgb_msg->height) {
    CameraInfo info_msg_tmp = *info_msg;
    info_msg_tmp.width = depth_msg->width;
    info_msg_tmp.height = depth_msg->height;
    float ratio = static_cast<float>(depth_msg->width) / static_cast<float>(rgb_msg->width);
    info_msg_tmp.k[0] *= ratio;
    info_msg_tmp.k[2] *= ratio;
    info_msg_tmp.k[4] *= ratio;
    info_msg_tmp.k[5] *= ratio;
    info_msg_tmp.p[0] *= ratio;
    info_msg_tmp.p[2] *= ratio;
    info_msg_tmp.p[5] *= ratio;
    info_msg_tmp.p[6] *= ratio;
    model_.fromCameraInfo(info_msg_tmp);

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(rgb_msg, rgb_msg->encoding);
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv_bridge::CvImage cv_rsz;
    cv_rsz.header = cv_ptr->header;
    cv_rsz.encoding = cv_ptr->encoding;
    cv::resize(
      cv_ptr->image.rowRange(0, depth_msg->height / ratio), cv_rsz.image,
      cv::Size(depth_msg->width, depth_msg->height));
    if ((rgb_msg->encoding == sensor_msgs::image_encodings::RGB8) ||
      (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) ||
      (rgb_msg->encoding == sensor_msgs::image_encodings::MONO8))
    {
      rgb_msg = cv_rsz.toImageMsg();
    } else {
      rgb_msg =
        cv_bridge::toCvCopy(cv_rsz.toImageMsg(), sensor_msgs::image_encodings::RGB8)->toImageMsg();
    }

    RCLCPP_ERROR(
      get_logger(), "Depth resolution (%ux%u) does not match RGB resolution (%ux%u)",
      depth_msg->width, depth_msg->height, rgb_msg->width, rgb_msg->height);
    return;
  } else {
    rgb_msg = rgb_msg_in;
  }

  // check if the input id image has to be resized
  Image::ConstSharedPtr id_msg = id_msg_in;
  if(depth_msg->width != id_msg->width || depth_msg->height != id_msg->height) {

    float ratio = static_cast<float>(depth_msg->width) / static_cast<float>(id_msg->width);

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(id_msg, id_msg->encoding);
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv_bridge::CvImage cv_rsz;
    cv_rsz.header = cv_ptr->header;
    cv_rsz.encoding = cv_ptr->encoding;
    cv::resize(
      cv_ptr->image.rowRange(0, depth_msg->height / ratio), cv_rsz.image,
      cv::Size(depth_msg->width, depth_msg->height));
    if ((id_msg->encoding == sensor_msgs::image_encodings::RGB8) ||
      (id_msg->encoding == sensor_msgs::image_encodings::BGR8) ||
      (id_msg->encoding == sensor_msgs::image_encodings::MONO8)) {
      id_msg = cv_rsz.toImageMsg();
    } else {
      id_msg =
        cv_bridge::toCvCopy(cv_rsz.toImageMsg(), sensor_msgs::image_encodings::MONO8)->toImageMsg();
    }

    RCLCPP_ERROR(
      get_logger(), "Depth resolution (%ux%u) does not match ID resolution (%ux%u)",
      depth_msg->width, depth_msg->height, id_msg->width, id_msg->height);
    return;
  } else {
    id_msg = id_msg_in;
  }

  // Supported color encodings: RGB8, BGR8, MONO8
  int red_offset, green_offset, blue_offset, color_step;
  // std::cout<<"rgb_msg encoding: "<<rgb_msg->encoding<<std::endl;
  if (rgb_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::RGBA8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 4;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 3;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGRA8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 4;
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    red_offset = 0;
    green_offset = 0;
    blue_offset = 0;
    color_step = 1;
  } else {
    try {
      rgb_msg = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8)->toImageMsg();
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(
        get_logger(), "Unsupported encoding [%s]: %s", rgb_msg->encoding.c_str(), e.what());
      return;
    }
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  }

  auto cloud_msg = std::make_shared<PointCloud2>();
  cloud_msg->header = depth_msg->header;  // Use depth image time stamp
  cloud_msg->height = depth_msg->height;
  cloud_msg->width = depth_msg->width;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
  // pcd_modifier.setPointCloud2FieldsByString(3, "xyz", "rgb", "label"); 
  // pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb" ); 

  pcd_modifier.setPointCloud2Fields(
  7,
  "x", 1, sensor_msgs::msg::PointField::FLOAT32,
  "y", 1, sensor_msgs::msg::PointField::FLOAT32,
  "z", 1, sensor_msgs::msg::PointField::FLOAT32,
  "r", 1, sensor_msgs::msg::PointField::UINT8,
  "g", 1, sensor_msgs::msg::PointField::UINT8,
  "b", 1, sensor_msgs::msg::PointField::UINT8,
  "label", 1, sensor_msgs::msg::PointField::UINT8);



  if (!filter_labels_.empty()) {
    // convert depth image to pointcloud with condition filter of label
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      convertDepthwithLabel<uint16_t>(depth_msg, cloud_msg, id_msg, filter_labels_, filter_keep_,model_);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      convertDepthwithLabel<float>(depth_msg, cloud_msg, id_msg, filter_labels_, filter_keep_,model_);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
      return;
    }
  }else{
      // Convert Depth Image to Pointcloud
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      convertDepth<uint16_t>(depth_msg, cloud_msg, model_);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      convertDepth<float>(depth_msg, cloud_msg, model_);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
      return;
    }
    //conver label
    if (id_msg->encoding == sensor_msgs::image_encodings::MONO8) {
      convertLabel(id_msg, cloud_msg);
    } else if (id_msg->encoding == sensor_msgs::image_encodings::MONO16) {
      convertLabel(id_msg, cloud_msg);
    } else if (id_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      convertLabel(id_msg, cloud_msg);
    } else if (id_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      convertLabel(id_msg, cloud_msg);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Intensity image has unsupported encoding [%s]",
        id_msg->encoding.c_str());
      return;
    }
  }

  // If configured, mask points based on label (set XYZ to NaN)
  // if (!filter_labels_.empty()) {
  //   const float bad_point = std::numeric_limits<float>::quiet_NaN();
  //   sensor_msgs::PointCloud2Iterator<uint8_t> iter_label(*cloud_msg, "label");
  //   sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  //   sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  //   sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  //   for (size_t v = 0; v < cloud_msg->height; ++v) {
  //     for (size_t u = 0; u < cloud_msg->width; ++u, ++iter_label, ++iter_x, ++iter_y, ++iter_z) {
  //       const uint8_t lbl = *iter_label;
  //       const bool in_set = (filter_labels_.find(lbl) != filter_labels_.end());
  //       const bool should_mask = filter_keep_ ? !in_set : in_set;
  //       if (should_mask) {
  //         *iter_x = bad_point;
  //         *iter_y = bad_point;
  //         *iter_z = bad_point;
  //       }
  //     }
  //   }
  // }
  

  // Convert RGB + label
  // if (rgb_msg->encoding == sensor_msgs::image_encodings::RGB8) {
  //   convertRgbLabel(rgb_msg, id_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  // } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
  //   convertRgbLabel(rgb_msg, id_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  // } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGRA8) {
  //   convertRgbLabel(rgb_msg, id_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  // } else if (rgb_msg->encoding == sensor_msgs::image_encodings::RGBA8) {
  //   convertRgbLabel(rgb_msg, id_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  // } else if (rgb_msg->encoding == sensor_msgs::image_encodings::MONO8) {
  //   convertRgbLabel(rgb_msg, id_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  // } else {
  //   RCLCPP_ERROR(
  //     get_logger(), "RGB image has unsupported encoding [%s]", rgb_msg->encoding.c_str());
  //   return;
  // }


  // //only convert rgb
    if (rgb_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    convertRgb(rgb_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    convertRgb(rgb_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::BGRA8) {
    convertRgb(rgb_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::RGBA8) {
    convertRgb(rgb_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  } else if (rgb_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    convertRgb(rgb_msg, cloud_msg, red_offset, green_offset, blue_offset, color_step);
  } else {
    RCLCPP_ERROR(
      get_logger(), "RGB image has unsupported encoding [%s]", rgb_msg->encoding.c_str());
    return;
  }

  // RCLCPP_INFO(
  //     get_logger(), "publishing point cloud with label channel, width: %d, height: %d",
  //     cloud_msg->width, cloud_msg->height);
  // RCLCPP_INFO(
  //     get_logger(), "cloud msg size: %d", static_cast<int>(cloud_msg->data.size()));
  pub_point_cloud_->publish(*cloud_msg);
}

}  // namespace depth_image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
RCLCPP_COMPONENTS_REGISTER_NODE(depth_image_proc::PointCloudXyzrgbLabelNode)
