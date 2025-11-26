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
  int queue_size = this->declare_parameter<int>("queue_size", 150);
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
      // sub_rgb_,
      // sub_id_,
      sub_combined_,
      sub_conf_,
      sub_info_);
    exact_sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4
      ));
  } else {
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(queue_size), sub_depth_, sub_combined_, sub_conf_, sub_info_); //sub_rgb_, sub_id_,
    sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4));
  }


  connectCb();

  std::lock_guard<std::mutex> lock(connect_mutex_);

  pub_point_cloud_ = create_publisher<PointCloud2>("points", rclcpp::SensorDataQoS());
  pub_ground_point_cloud_ = create_publisher<PointCloud2>("ground_points", rclcpp::SensorDataQoS());

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
    // sub_rgb_.unsubscribe();
    // sub_id_.unsubscribe();
    sub_combined_.unsubscribe();
    sub_conf_.unsubscribe();
    sub_info_.unsubscribe();
  } else if (!sub_depth_.getSubscriber()) {
    // parameter for depth_image_transport hint
    std::string depth_image_transport_param = "depth_image_transport";
    image_transport::TransportHints depth_hints(this, "raw", depth_image_transport_param);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.qos_overriding_options = rclcpp::QosOverridingOptions {
      {
        rclcpp::QosPolicyKind::Depth,
        rclcpp::QosPolicyKind::Durability,
        rclcpp::QosPolicyKind::History,
        rclcpp::QosPolicyKind::Reliability,
      }};

    sub_depth_.subscribe(
      this, "depth_registered/image_rect",
      depth_hints.getTransport(), rmw_qos_profile_default, sub_opts);


    sub_info_.subscribe(this, "rgb/camera_info");

    image_transport::TransportHints combined_hints(this, "raw");
    sub_combined_.subscribe(
        this, "combined/image_rect_combined",
        combined_hints.getTransport(), rmw_qos_profile_default, sub_opts);

    //confidence uses confidence ros transport hints.
    image_transport::TransportHints conf_hints(this, "raw");
    sub_conf_.subscribe(
        this, "confidence/image_rect_confidence",
        conf_hints.getTransport(), rmw_qos_profile_default, sub_opts);

  }
}

void PointCloudXyzrgbLabelNode::imageCb(
  const Image::ConstSharedPtr & depth_msg,
  const Image::ConstSharedPtr & combined_msg_in,
  const Image::ConstSharedPtr & conf_msg_in,
  const CameraInfo::ConstSharedPtr & info_msg)
{
    // RCLCPP_INFO(
    //   get_logger(), "PointCloudXyzrgbLabelNode::imageCb called");
  //check for bad inputs of confidence image
  if (depth_msg->header.frame_id != conf_msg_in->header.frame_id) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      10000,  // 10 seconds
      "Depth image frame id [%s] doesn't match Confidence image frame id [%s]",
      depth_msg->header.frame_id.c_str(), conf_msg_in->header.frame_id.c_str());
  }

  // Update camera model
  model_.fromCameraInfo(info_msg);

  // Check if the input color image has to be resized
  Image::ConstSharedPtr combined_msg = combined_msg_in;
  if (depth_msg->width != combined_msg->width || depth_msg->height != combined_msg->height) {
    throw std::runtime_error("Combined segmentation image size does not match depth image size.");
    return;
  } else {
    combined_msg = combined_msg_in;
  }

  // check if the input color image has to be resized
  Image::ConstSharedPtr conf_msg = conf_msg_in;
  if(depth_msg->width != conf_msg->width || depth_msg->height != conf_msg->height) {
    throw std::runtime_error("Confidence image size does not match depth image size.");
    return;
  } else {
    conf_msg = conf_msg_in;
  }

  // Supported color encodings: RGB8, BGR8, MONO8
  int red_offset, green_offset, blue_offset, color_step;
  // std::cout<<"rgb_msg encoding: "<<rgb_msg->encoding<<std::endl;
  if (combined_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 3;
  } else if (combined_msg->encoding == sensor_msgs::image_encodings::RGBA8) {
    red_offset = 0;
    green_offset = 1;
    blue_offset = 2;
    color_step = 4;
  } else if (combined_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 3;
  } else if (combined_msg->encoding == sensor_msgs::image_encodings::BGRA8) {
    red_offset = 2;
    green_offset = 1;
    blue_offset = 0;
    color_step = 4;
  } else if (combined_msg->encoding == sensor_msgs::image_encodings::MONO8) {
    red_offset = 0;
    green_offset = 0;
    blue_offset = 0;
    color_step = 1;
  } else {
    throw std::runtime_error("Unsupported combined image encoding: " + combined_msg->encoding);
    return;
    try {
      combined_msg = cv_bridge::toCvCopy(combined_msg, sensor_msgs::image_encodings::RGB8)->toImageMsg();
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(
        get_logger(), "Unsupported encoding [%s]: %s", combined_msg->encoding.c_str(), e.what());
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
  pcd_modifier.setPointCloud2Fields(
  5,
  "x", 1, sensor_msgs::msg::PointField::FLOAT32,
  "y", 1, sensor_msgs::msg::PointField::FLOAT32,
  "z", 1, sensor_msgs::msg::PointField::FLOAT32,
  "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
  "label", 1, sensor_msgs::msg::PointField::UINT8);
  pcd_modifier.resize(static_cast<uint32_t>(cloud_msg->width) * static_cast<uint32_t>(cloud_msg->height));


  auto ground_cloud_msg = std::make_shared<PointCloud2>();
  ground_cloud_msg->header = depth_msg->header;  // Use depth image time stamp
  ground_cloud_msg->height = depth_msg->height;
  ground_cloud_msg->width = depth_msg->width;
  ground_cloud_msg->is_dense = false;
  ground_cloud_msg->is_bigendian = false;
  sensor_msgs::PointCloud2Modifier ground_pcd_modifier(*ground_cloud_msg);
  ground_pcd_modifier.setPointCloud2Fields(
  5,
  "x", 1, sensor_msgs::msg::PointField::FLOAT32,
  "y", 1, sensor_msgs::msg::PointField::FLOAT32,
  "z", 1, sensor_msgs::msg::PointField::FLOAT32,
  "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
  "label", 1, sensor_msgs::msg::PointField::UINT8);
  ground_pcd_modifier.resize(static_cast<uint32_t>(ground_cloud_msg->width) * static_cast<uint32_t>(ground_cloud_msg->height));


  if (!filter_labels_.empty()) {
    // convert depth image to pointcloud with condition filter of label
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      std::cout<<"convertDepthwithLabelAndConfidence - uint16_t"<<std::endl;
      // publish ground cloud for octomap update in case missegmentation of internimage 
      convertDepthwithCombinedmsg<uint16_t>(depth_msg, cloud_msg, combined_msg, conf_msg, filter_labels_, filter_keep_, model_, red_offset, green_offset, blue_offset, color_step);
      convertDepthwithCombinedmsg<uint16_t>(depth_msg, cloud_msg, combined_msg, conf_msg, filter_labels_, !filter_keep_, model_, red_offset, green_offset, blue_offset, color_step);
    } else if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      // accept combined msg in case color msg and id msg are not well aligned
      convertDepthwithCombinedmsg<float>(depth_msg, cloud_msg, combined_msg, conf_msg, filter_labels_, filter_keep_, model_, red_offset, green_offset, blue_offset, color_step);
      convertDepthwithCombinedmsg<float>(depth_msg, ground_cloud_msg, combined_msg, conf_msg, filter_labels_, !filter_keep_, model_, red_offset, green_offset, blue_offset, color_step);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Depth image has unsupported encoding [%s]", depth_msg->encoding.c_str());
      return;
    }
  }else{
    throw std::runtime_error("No filter labels specified, full pointcloud generation not implemented yet.");
    return;

  }

  pub_point_cloud_->publish(*cloud_msg);
  pub_ground_point_cloud_->publish(*ground_cloud_msg);
}

}  // namespace depth_image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
RCLCPP_COMPONENTS_REGISTER_NODE(depth_image_proc::PointCloudXyzrgbLabelNode)
