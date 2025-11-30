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
#include <message_filters/simple_filter.h>




namespace depth_image_proc
{


PointCloudXyzrgbLabelNode::PointCloudXyzrgbLabelNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("PointCloudXyzrgbLabelNode", options)
{
  // Read parameters
  int queue_size = this->declare_parameter<int>("queue_size", 20);
  bool use_exact_sync = this->declare_parameter<bool>("exact_sync", true);
  // Slop (max interval duration) for ApproximateTime sync in seconds
  double approx_sync_slop = this->declare_parameter<double>("approx_sync_slop", 0.5);

  std::string target_frame = this->declare_parameter<std::string>("target_frame", "zed_left_camera_optical_frame");

  // Target frame for transforming incoming lidar pointcloud
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Label filtering parameters
  // filter_labels: list of integer labels. By default, points with these labels will be dropped (masked as NaN).
  // Set filter_keep=true to invert behavior and keep only points with these labels.
  auto filter_labels_param = this->declare_parameter<std::vector<int64_t>>("filter_labels", std::vector<int64_t>{});
  filter_labels_param = this->get_parameter("filter_labels").as_integer_array();
  filter_keep_ = this->declare_parameter<bool>("filter_keep", false);
  filter_keep_ = this->get_parameter("filter_keep").as_bool();
  target_frame_ = this->get_parameter("target_frame").as_string();

  this->declare_parameter<int>("outlier_reject_MeanK", 50);
  this->declare_parameter<double>("outlier_reject_StddevMulThresh", 1.0);
  outlier_reject_MeanK_ = this->get_parameter("outlier_reject_MeanK").as_int();
  outlier_reject_StddevMulThresh_ = this->get_parameter("outlier_reject_StddevMulThresh").as_double();

  

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
      sub_combined_,
      sub_info_);
    exact_sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
      ));
  } else {
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(queue_size), sub_depth_, sub_combined_, sub_info_); //sub_rgb_, sub_id_,
    // Configure slop window for ApproximateTime policy
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(approx_sync_slop));
    sync_->registerCallback(
      std::bind(
        &PointCloudXyzrgbLabelNode::imageCb,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));
  }

  lidar_cache_ = std::make_shared<PointCloudCache>();
  

  using SimpleFilter = message_filters::SimpleFilter<PointCloud2>;
  auto dummy_filter = std::make_shared<SimpleFilter>();
  lidar_cache_ = std::make_shared<PointCloudCache>(*dummy_filter, LIDAR_CACHE_SIZE);

  connectCb();

  std::lock_guard<std::mutex> lock(connect_mutex_);

  pub_point_cloud_ = create_publisher<PointCloud2>("points", rclcpp::SensorDataQoS());
  pub_ground_point_cloud_ = create_publisher<PointCloud2>("ground_points", rclcpp::SensorDataQoS());



}

// Handles (un)subscribing when clients (un)subscribe
void PointCloudXyzrgbLabelNode::connectCb()
{
  RCLCPP_INFO(
    get_logger(), "PointCloudXyzrgbLabelNode::connectCb called");
  std::lock_guard<std::mutex> lock(connect_mutex_);
  // TODO(ros2) Implement getNumSubscribers when rcl/rmw support it
  // if (pub_point_cloud_->getNumSubscribers() == 0)
  if (0) {
    // TODO(ros2) Implement getNumSubscribers when rcl/rmw support it
    sub_depth_.unsubscribe();
    sub_combined_.unsubscribe();
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

    if (!sub_pointcloud_cache_) {
        rclcpp::SubscriptionOptions sub_opts;
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)); 
        qos.reliable();                             
        sub_pointcloud_cache_ = create_subscription<PointCloud2>( 
        "lidar/points",
        qos,
        [this](const PointCloud2::SharedPtr msg) {
          // RCLCPP_INFO(this->get_logger(), "LiDAR PointCloud received and added to cache: %s", msg->header.frame_id.c_str());
          lidar_cache_->add(std::static_pointer_cast<const PointCloud2>(msg));
        },
        sub_opts);
    }

  }
}

void PointCloudXyzrgbLabelNode::imageCb(
  const Image::ConstSharedPtr & depth_msg,
  const Image::ConstSharedPtr & combined_msg_in,
  const CameraInfo::ConstSharedPtr & info_msg)
{
  // std::cout<<"PointCloudXyzrgbLabelNode::imageCb called"<<std::endl;
  rclcpp::Time combined_time(combined_msg_in->header.stamp);
  PointCloud2::ConstSharedPtr current_lidar_pointcloud = lidar_cache_->getElemBeforeTime(combined_time);

  if (!current_lidar_pointcloud) {
    RCLCPP_WARN(this->get_logger(), "No synchronized LiDAR pointcloud found in cache.");
    return;
  }
  PointCloud2::ConstSharedPtr latest_lidar_pointcloud_ = current_lidar_pointcloud;

  // rclcpp::Time lidar_time(latest_lidar_pointcloud_->header.stamp);
  // double time_diff = (combined_time - lidar_time).seconds(); 
  // std::cout << "latest_lidar_pointcloud_ stamp: " << lidar_time.nanoseconds() << std::endl;
  // std::cout << "combined_msg_in stamp: " << combined_time.nanoseconds() << std::endl;
  // std::cout << "Time Difference (Combined - LiDAR) [s]: " << time_diff << std::endl;
  // std::cout << "----------------------------" << std::endl;

  if (latest_lidar_pointcloud_->header.frame_id != target_frame_) {
    try {
      geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
        target_frame_,
        latest_lidar_pointcloud_->header.frame_id,
        rclcpp::Time(combined_msg_in->header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      sensor_msgs::msg::PointCloud2 transformed;
      tf2::doTransform(*latest_lidar_pointcloud_, transformed, t);
      transformed.header.frame_id = target_frame_;
      latest_transformed_pointcloud_ = std::make_shared<sensor_msgs::msg::PointCloud2>(std::move(transformed));
      RCLCPP_DEBUG(this->get_logger(), "Transformed pointcloud from %s to %s",
                   latest_lidar_pointcloud_->header.frame_id.c_str(), target_frame_.c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(),
                  "TF lookup/transform failed: %s. Dropping this LiDAR cloud (frame: %s); waiting for transform to %s",
                  e.what(), latest_lidar_pointcloud_->header.frame_id.c_str(), target_frame_.c_str());
      // Do not update latest_transformed_pointcloud_ to avoid projecting from a non-optical frame
    }
  } else {
    latest_transformed_pointcloud_ = 
      std::make_shared<sensor_msgs::msg::PointCloud2>(*latest_lidar_pointcloud_); 
}
  // RCLCPP_INFO(
  //   get_logger(), "PointCloudXyzrgbLabelNode::imageCb called");

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
    // try {
    //   combined_msg = cv_bridge::toCvCopy(combined_msg, sensor_msgs::image_encodings::RGB8)->toImageMsg();
    // } catch (cv_bridge::Exception & e) {
    //   RCLCPP_ERROR(
    //     get_logger(), "Unsupported encoding [%s]: %s", combined_msg->encoding.c_str(), e.what());
    //   return;
    // }
    // red_offset = 0;
    // green_offset = 1;
    // blue_offset = 2;
    // color_step = 3;
  }

  auto cloud_msg = std::make_shared<PointCloud2>();
  cloud_msg->header = latest_transformed_pointcloud_->header;  // Use depth image time stamp
  cloud_msg->height = latest_transformed_pointcloud_->height;
  cloud_msg->width = latest_transformed_pointcloud_->width;
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

  auto ground_cloud_msg = std::make_shared<PointCloud2>();
  ground_cloud_msg->header = latest_transformed_pointcloud_->header;  // Use depth image time stamp
  ground_cloud_msg->height = latest_transformed_pointcloud_->height;
  ground_cloud_msg->width = latest_transformed_pointcloud_->width;
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



  if (!filter_labels_.empty()) {
    if(latest_transformed_pointcloud_ == nullptr){
      RCLCPP_ERROR(
      get_logger(), "No LiDAR pointcloud received yet, cannot proceed.");
      return;
    }
    if(latest_transformed_pointcloud_->header.stamp != last_processed_lidar_cloud_stamp_){
      PointCloud2::ConstSharedPtr pointcloud_ptr = latest_transformed_pointcloud_;
      convertLabelAndRgbWithLidar<float>(cloud_msg, combined_msg, pointcloud_ptr, filter_labels_, filter_keep_, model_, red_offset, green_offset, blue_offset);
      convertLabelAndRgbWithLidar<float>(ground_cloud_msg, combined_msg, pointcloud_ptr, filter_labels_, !filter_keep_, model_, red_offset, green_offset, blue_offset);
      last_processed_lidar_cloud_stamp_ = pointcloud_ptr->header.stamp;
    }
  }else{
    throw std::runtime_error("No filter labels specified, full pointcloud generation not implemented yet.");
    return;
    
  }




  // pcl point with labels outlier removal
  PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*cloud_msg, pcl_cloud);
  PCLPointCloud2::Ptr input_pcl_cloud_points = std::make_shared<PCLPointCloud2>(std::move(pcl_cloud));
  PCLPointCloud2::Ptr output_pcl_cloud_points = std::make_shared<pcl::PCLPointCloud2>();

  pcl::StatisticalOutlierRemoval<PCLPointCloud2> sor;
  sor.setInputCloud (input_pcl_cloud_points);
  sor.setMeanK (outlier_reject_MeanK_);
  sor.setStddevMulThresh (outlier_reject_StddevMulThresh_);
  sor.filter (*output_pcl_cloud_points);

  // Convert filtered PCLPointCloud2 back to ROS PointCloud2
  PointCloud2::Ptr filtered_cloud_msg = std::make_shared<PointCloud2>();
  pcl_conversions::fromPCL(*output_pcl_cloud_points, *filtered_cloud_msg);
  filtered_cloud_msg->header = cloud_msg->header;

  PCLPointCloud2 ground_pcl_cloud;
  pcl_conversions::toPCL(*ground_cloud_msg, ground_pcl_cloud);
  PCLPointCloud2::Ptr input_ground_pcl_cloud_points = std::make_shared<PCLPointCloud2>(std::move(ground_pcl_cloud));
  PCLPointCloud2::Ptr output_ground_pcl_cloud_points = std::make_shared<pcl::PCLPointCloud2>();

  pcl::StatisticalOutlierRemoval<PCLPointCloud2> ground_sor;
  ground_sor.setInputCloud (input_ground_pcl_cloud_points);
  ground_sor.setMeanK (outlier_reject_MeanK_);
  ground_sor.setStddevMulThresh (outlier_reject_StddevMulThresh_);
  ground_sor.filter (*output_ground_pcl_cloud_points);

  // Convert filtered PCLPointCloud2 back to ROS PointCloud2
  PointCloud2::Ptr filtered_ground_cloud_msg = std::make_shared<PointCloud2>();
  pcl_conversions::fromPCL(*output_ground_pcl_cloud_points, *filtered_ground_cloud_msg);
  filtered_ground_cloud_msg->header = ground_cloud_msg->header;

  pub_point_cloud_->publish(*filtered_cloud_msg);
  pub_ground_point_cloud_->publish(*filtered_ground_cloud_msg);


  // pub_point_cloud_->publish(*cloud_msg);
  // pub_ground_point_cloud_->publish(*ground_cloud_msg);
}

}  // namespace depth_image_proc

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
RCLCPP_COMPONENTS_REGISTER_NODE(depth_image_proc::PointCloudXyzrgbLabelNode)