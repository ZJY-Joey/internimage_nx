#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
// tf2 core transform function template
#include <tf2/transform_datatypes.h>

#include <deque>
#include <memory>

using std::placeholders::_1;

namespace depth_cloud_acc
{

class DepthCloudAccNode : public rclcpp::Node
{
public:
  DepthCloudAccNode()
  : Node("DepthCloudAccNode")
  {
    this->declare_parameter<std::string>("input_depth_points_topic", "/internimage/segmentation/filtered/points");
    this->declare_parameter<std::string>("output_depth_points_topic", "/internimage/segmentation/acc_global_map");
    this->declare_parameter<std::string>("fixed_frame", "world");
    this->declare_parameter<double>("publish_period", 0.1);
    this->declare_parameter<int>("max_clouds", 100);
    this->declare_parameter<bool>("enable_transform", true);
    this->declare_parameter<double>("lookup_timeout", 0.5);
    this->declare_parameter<int>("max_aggregated_points", 1000000); // hard cap to prevent memory blowup
    this->declare_parameter<int>("rebuild_keep_last_clouds", 50); // when rebuilding, number of recent clouds to retain
    this->declare_parameter<std::string>("aggregation_strategy", "rebuild"); // or 'truncate'
    this->declare_parameter<int>("log_every_n", 20);

    input_topic_ = this->get_parameter("input_depth_points_topic").as_string();
    output_topic_ = this->get_parameter("output_depth_points_topic").as_string();
    fixed_frame_ = this->get_parameter("fixed_frame").as_string();
    publish_period_ = this->get_parameter("publish_period").as_double();
    max_clouds_ = this->get_parameter("max_clouds").as_int();
    enable_transform_ = this->get_parameter("enable_transform").as_bool();
    lookup_timeout_ = this->get_parameter("lookup_timeout").as_double();
  max_aggregated_points_ = this->get_parameter("max_aggregated_points").as_int();
  rebuild_keep_last_clouds_ = this->get_parameter("rebuild_keep_last_clouds").as_int();
  aggregation_strategy_ = this->get_parameter("aggregation_strategy").as_string();
  log_every_n_ = this->get_parameter("log_every_n").as_int();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // construct TransformListener with node pointer
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

    // subscription with sensor data QoS
    rclcpp::SensorDataQoS qos;
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&DepthCloudAccNode::pointcloud_callback, this, _1));

  // publisher reliable
  rclcpp::QoS pub_qos(10);
  pub_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
  publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, pub_qos);

    // timer (milliseconds)
    auto period_ms = std::chrono::milliseconds(static_cast<int>(publish_period_ * 1000.0));
    timer_ = this->create_wall_timer(period_ms, std::bind(&DepthCloudAccNode::publish_map, this));

    // reset service
    reset_srv_ = this->create_service<std_srvs::srv::Empty>("/reset_map",
      std::bind(&DepthCloudAccNode::handle_reset, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "DepthCloudAccNode started: sub='%s' pub='%s' frame='%s' period=%.3f strategy=%s max_agg=%d",
      input_topic_.c_str(), output_topic_.c_str(), fixed_frame_.c_str(), publish_period_, aggregation_strategy_.c_str(), max_aggregated_points_);
  }

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud = msg;
    if (enable_transform_ && msg->header.frame_id != fixed_frame_) {
      try {
        geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
          fixed_frame_, msg->header.frame_id, rclcpp::Time(msg->header.stamp),
          rclcpp::Duration::from_seconds(lookup_timeout_));
  sensor_msgs::msg::PointCloud2 transformed;
  // Use generic tf2::doTransform (PointCloud2 specialization provided by tf2_sensor_msgs header)
  tf2::doTransform(*msg, transformed, t);
        transformed.header.frame_id = fixed_frame_;
        cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(transformed);
        RCLCPP_DEBUG(this->get_logger(), "Transformed cloud from %s to %s", msg->header.frame_id.c_str(), fixed_frame_.c_str());
      } catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "TF lookup/transform failed: %s", e.what());
        // fall back to original
      }
    }

    // append to deque and aggregated
    if (max_clouds_ > 0) {
      if ((int)clouds_.size() >= max_clouds_) {
        clouds_.pop_front();
      }
      clouds_.push_back(cloud);
    }
    append_to_aggregated(cloud);
    if (++received_count_ % std::max(1, log_every_n_) == 0) {
      RCLCPP_INFO(this->get_logger(), "Received %d clouds | aggregated_points=%u", received_count_, aggregated_ ? aggregated_->width : 0);
    }
  }

  void append_to_aggregated(const sensor_msgs::msg::PointCloud2::SharedPtr & cloud)
  {
    if (!aggregated_) {
      aggregated_ = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
      // ensure mutable data buffer
      aggregated_->data = cloud->data;
      return;
    }

    // basic compatibility checks
    if (cloud->point_step != aggregated_->point_step ||
        cloud->is_bigendian != aggregated_->is_bigendian ||
        cloud->fields != aggregated_->fields ||
        cloud->height != aggregated_->height) {
      RCLCPP_WARN(this->get_logger(), "Incoming cloud layout mismatch; skipping accumulation.");
      return;
    }

    // append data
    aggregated_->data.insert(aggregated_->data.end(), cloud->data.begin(), cloud->data.end());
    aggregated_->width += cloud->width;
    aggregated_->row_step = aggregated_->width * aggregated_->point_step;
    aggregated_->header.frame_id = fixed_frame_;
    enforce_limits();
  }

  void publish_map()
  {
    if (aggregated_) {
      aggregated_->header.stamp = this->now();
      publisher_->publish(*aggregated_);
    }
  }

  void handle_reset(const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/,
                    std::shared_ptr<std_srvs::srv::Empty::Response> /*res*/)
  {
    clouds_.clear();
    aggregated_.reset();
    received_count_ = 0;
    RCLCPP_INFO(this->get_logger(), "Map reset: cleared aggregated cloud and stored clouds.");
  }

  void enforce_limits()
  {
    if (!aggregated_ || max_aggregated_points_ <= 0) return;
    if (static_cast<int>(aggregated_->width) <= max_aggregated_points_) return;

    if (aggregation_strategy_ == "truncate") {
      // Truncate newest excess points: shrink data vector to cap.
      int point_step = aggregated_->point_step;
      int excess = static_cast<int>(aggregated_->width) - max_aggregated_points_;
      size_t bytes_excess = static_cast<size_t>(excess) * point_step;
      if (bytes_excess < aggregated_->data.size()) {
        aggregated_->data.resize(aggregated_->data.size() - bytes_excess);
        aggregated_->width = max_aggregated_points_;
        aggregated_->row_step = aggregated_->width * point_step;
        RCLCPP_WARN(this->get_logger(), "Truncated aggregated cloud to %d points (strategy=truncate)", max_aggregated_points_);
      }
      return;
    }

    // Default: rebuild from tail of deque
    if (clouds_.empty()) return; // should not happen
    auto rebuilt = std::make_shared<sensor_msgs::msg::PointCloud2>();
    bool base_set = false;
    int kept = 0;
    for (auto it = clouds_.rbegin(); it != clouds_.rend() && kept < rebuild_keep_last_clouds_; ++it) {
      const auto & c = *it;
      if (!base_set) {
        *rebuilt = *c;
        rebuilt->data = c->data;
        base_set = true;
      } else {
        if (c->point_step != rebuilt->point_step || c->is_bigendian != rebuilt->is_bigendian || c->fields != rebuilt->fields || c->height != rebuilt->height) {
          // skip incompatible older cloud
          continue;
        }
        rebuilt->data.insert(rebuilt->data.end(), c->data.begin(), c->data.end());
        rebuilt->width += c->width;
        rebuilt->row_step = rebuilt->width * rebuilt->point_step;
      }
      ++kept;
      if (static_cast<int>(rebuilt->width) >= max_aggregated_points_) break; // reached cap
    }
    aggregated_ = rebuilt;
    RCLCPP_WARN(this->get_logger(), "Rebuilt aggregated cloud from last %d clouds -> %u points (cap=%d)", kept, aggregated_->width, max_aggregated_points_);
  }

  // parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string fixed_frame_;
  double publish_period_;
  int max_clouds_;
  bool enable_transform_;
  double lookup_timeout_;
  int max_aggregated_points_;
  int rebuild_keep_last_clouds_;
  std::string aggregation_strategy_;
  int log_every_n_;
  int received_count_{0};

  // tf
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

  // data
  std::deque<sensor_msgs::msg::PointCloud2::SharedPtr> clouds_;
  sensor_msgs::msg::PointCloud2::SharedPtr aggregated_;
};



}// namespace depth_cloud_acc

// Standalone executable entry point (since we are not building a component library here)
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<depth_cloud_acc::DepthCloudAccNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


// #include "rclcpp_components/register_node_macro.hpp"

// // Register the component with class_loader.
// RCLCPP_COMPONENTS_REGISTER_NODE(depth_cloud_acc::DepthCloudAccNode)



