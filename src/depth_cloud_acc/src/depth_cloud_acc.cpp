#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
// tf2 core transform function template
#include <tf2/transform_datatypes.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <deque>
#include <memory>
#include <mutex>

using std::placeholders::_1;

namespace depth_cloud_acc
{

bool isFieldsEqual(
  const std::vector<pcl::PCLPointField> & fields1,
  const std::vector<pcl::PCLPointField> & fields2)
{
  if (fields1.size() != fields2.size()) return false;
  for (size_t i = 0; i < fields1.size(); ++i) {
    if (fields1[i].name != fields2[i].name ||
        fields1[i].offset != fields2[i].offset ||
        fields1[i].datatype != fields2[i].datatype ||
        fields1[i].count != fields2[i].count) {
      return false;
    }
  }
  return true;
}


class DepthCloudAccNode : public rclcpp::Node
{
public:
  DepthCloudAccNode()
  : Node("DepthCloudAccNode")
  {
    this->declare_parameter<std::string>("input_depth_points_topic", "/internimage/segmentation/filtered/points");
    this->declare_parameter<std::string>("output_depth_points_topic", "/internimage/segmentation/acc_global_map");
    this->declare_parameter<std::string>("fixed_frame", "world");
    this->declare_parameter<std::string>("robot_frame", "aliengo");
    this->declare_parameter<double>("publish_period", 0.1);
    this->declare_parameter<bool>("enable_transform", true);
    this->declare_parameter<double>("lookup_timeout", 0.5);
    this->declare_parameter<int>("log_every_n", 20);
    this->declare_parameter<bool>("acc_cloud_registered", false);
    this->declare_parameter<double>("voxel_leaf_size", 0.1);
    this->declare_parameter<int>("max_points", 500000);
    // passthrough limits relative to robot_frame
    this->declare_parameter<double>("pass_x_min", -10.0);
    this->declare_parameter<double>("pass_x_max", 10.0);
    this->declare_parameter<double>("pass_y_min", -10.0);
    this->declare_parameter<double>("pass_y_max", 10.0);
    this->declare_parameter<double>("pass_z_min", -2.0);
    this->declare_parameter<double>("pass_z_max", 2.0);

    input_topic_ = this->get_parameter("input_depth_points_topic").as_string();
    output_topic_ = this->get_parameter("output_depth_points_topic").as_string();
    fixed_frame_ = this->get_parameter("fixed_frame").as_string();
    robot_frame_ = this->get_parameter("robot_frame").as_string();
    publish_period_ = this->get_parameter("publish_period").as_double();
    enable_transform_ = this->get_parameter("enable_transform").as_bool();
    lookup_timeout_ = this->get_parameter("lookup_timeout").as_double();
    log_every_n_ = this->get_parameter("log_every_n").as_int();
    acc_cloud_registered_ = this->get_parameter("acc_cloud_registered").as_bool();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    max_points_ = this->get_parameter("max_points").as_int();

    // Validate parameters for safety
    if (voxel_leaf_size_ <= 0.0) {
      RCLCPP_WARN(this->get_logger(), "voxel_leaf_size must be > 0. Resetting to 0.1");
      voxel_leaf_size_ = 0.1;
    }
    if (max_points_ <= 0) {
      RCLCPP_WARN(this->get_logger(), "max_points must be > 0. Resetting to 500000");
      max_points_ = 500000;
    }
    if (pass_x_min_ > pass_x_max_) std::swap(pass_x_min_, pass_x_max_);
    if (pass_y_min_ > pass_y_max_) std::swap(pass_y_min_, pass_y_max_);
    if (pass_z_min_ > pass_z_max_) std::swap(pass_z_min_, pass_z_max_);

    pass_x_min_ = this->get_parameter("pass_x_min").as_double();
    pass_x_max_ = this->get_parameter("pass_x_max").as_double();
    pass_y_min_ = this->get_parameter("pass_y_min").as_double();
    pass_y_max_ = this->get_parameter("pass_y_max").as_double();
    pass_z_min_ = this->get_parameter("pass_z_min").as_double();
    pass_z_max_ = this->get_parameter("pass_z_max").as_double();

    // RCLCPP_INFO(this->get_logger(), "PassThrough limits set to: x:[%.2f, %.2f], y:[%.2f, %.2f], z:[%.2f, %.2f]",
    //   pass_x_min_, pass_x_max_, pass_y_min_, pass_y_max_, pass_z_min_, pass_z_max_);
    // RCLCPP_INFO(this->get_logger(),"input topic : %s", input_topic_.c_str());
    // RCLCPP_INFO(this->get_logger(),"output topic : %s", output_topic_.c_str());
      // throw std::runtime_error("PassThrough limits invalid.");
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

    RCLCPP_INFO(this->get_logger(), "DepthCloudAccNode started: sub='%s' pub='%s' fixed='%s' robot='%s' period=%.3f",
      input_topic_.c_str(), output_topic_.c_str(), fixed_frame_.c_str(), robot_frame_.c_str(), publish_period_);
  }

private:
  void filter_aggregated_in_robot_frame()
  {
    std::lock_guard<std::mutex> lock(agg_mutex_);
    if (!aggregated_ || aggregated_->width == 0) return;
    try {
      // Get robot position (translation only) in fixed_frame
      double rx = 0.0, ry = 0.0, rz = 0.0;
      if (enable_transform_) {
        geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
          fixed_frame_, robot_frame_, this->now(), rclcpp::Duration::from_seconds(lookup_timeout_));
        rx = t.transform.translation.x;
        ry = t.transform.translation.y;
        rz = t.transform.translation.z;
      }
      RCLCPP_INFO(this->get_logger(), "Filtering aggregated cloud with %u points", aggregated_ ? aggregated_->width : 0);
      pcl::PCLPointCloud2::Ptr input(new pcl::PCLPointCloud2(*aggregated_));
      pcl::PCLPointCloud2::Ptr output(new pcl::PCLPointCloud2());
      pcl::PassThrough<pcl::PCLPointCloud2> pass;
      pass.setInputCloud(input);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(rx + pass_x_min_, rx + pass_x_max_);
      pass.filter(*output);
      input.reset(new pcl::PCLPointCloud2(*output));
      pass.setInputCloud(input);
      pass.setFilterFieldName("y");
      pass.setFilterLimits(ry + pass_y_min_, ry + pass_y_max_);
      pass.filter(*output);
      input.reset(new pcl::PCLPointCloud2(*output));
      pass.setInputCloud(input);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(rz + pass_z_min_, rz + pass_z_max_);
      pass.filter(*output);
      input.reset(new pcl::PCLPointCloud2(*output));
      output->header.frame_id = fixed_frame_;
      aggregated_.reset(new pcl::PCLPointCloud2(*output));
      // RCLCPP_INFO(this->get_logger(), "Filtered aggregated cloud to width=%u", aggregated_->width);
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(this->get_logger(), "Filtering aggregated failed (TF): %s", e.what());
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Filtering aggregated failed: %s", e.what());
    }
  }

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // RCLCPP_INFO(this->get_logger(), "Received point cloud with %u points", msg->width);
    sensor_msgs::msg::PointCloud2::SharedPtr cloud = msg;
    if (enable_transform_ && msg->header.frame_id != fixed_frame_) {
      try {
        // RCLCPP_INFO(this->get_logger(), "msg-> header.stamp.sec: %u", msg->header.stamp.sec);
        geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
          fixed_frame_, msg->header.frame_id, rclcpp::Time(msg->header.stamp),
          rclcpp::Duration::from_seconds(lookup_timeout_));
        // RCLCPP_INFO(this->get_logger(), "Transform found from %s to %s",
                // msg->header.frame_id.c_str(), fixed_frame_.c_str());
        sensor_msgs::msg::PointCloud2 transformed;
        // Use generic tf2::doTransform (PointCloud2 specialization provided by tf2_sensor_msgs header)
        tf2::doTransform(*msg, transformed, t);
        // RCLCPP_INFO(this->get_logger(), "PointCloud2 transformed to %s frame", fixed_frame_.c_str());
        transformed.header.frame_id = fixed_frame_;
        cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(transformed);
        RCLCPP_DEBUG(this->get_logger(), "Transformed cloud from %s to %s", msg->header.frame_id.c_str(), fixed_frame_.c_str());
      } catch (const tf2::TransformException & e) {
        RCLCPP_WARN(this->get_logger(), "TF lookup/transform failed: %s", e.what());
        // fall back to original
      }
    }
    // RCLCPP_INFO(this->get_logger(), "Adding cloud with %u points to aggregation", cloud->width);

    // append to deque and aggregated
    // Convert incoming cloud to PCLPointCloud2 for internal storage
    pcl::PCLPointCloud2::Ptr pcl_cloud(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*cloud, *pcl_cloud);
    pcl_cloud->header.frame_id = fixed_frame_;
    append_to_aggregated(pcl_cloud);
    if (++received_count_ % std::max(1, log_every_n_) == 0) {
      // RCLCPP_INFO(this->get_logger(), "Received %d clouds | aggregated_points=%u", received_count_, aggregated_ ? aggregated_->width : 0);
    }
  }

  void append_to_aggregated(const pcl::PCLPointCloud2::Ptr & cloud)
  {
    if (!cloud) return;
    // Basic layout sanity check
    if (cloud->point_step == 0) {
      RCLCPP_WARN(this->get_logger(), "Incoming cloud point_step is 0. Ignoring.");
      return;
    }

    std::lock_guard<std::mutex> lock(agg_mutex_);
    // for (const auto &field : cloud->fields) {
    //   std::cout << field.name << " " << field.offset << " " << field.datatype << " " << field.count << "\n";
    // }
    // std::cout << "--------------------------------------------------------" << std::endl;
    if (!aggregated_) {
      aggregated_.reset(new pcl::PCLPointCloud2());
      aggregated_->fields = cloud->fields;
      aggregated_->is_bigendian = cloud->is_bigendian;
      aggregated_->point_step = cloud->point_step;
      aggregated_->height = 1;
      aggregated_->width = 0;
      aggregated_->row_step = 0;
      aggregated_->header.frame_id = fixed_frame_;
      aggregated_->data.clear();
      RCLCPP_DEBUG(this->get_logger(), "Initialized aggregated cloud with %zu fields", aggregated_->fields.size());
    } else if (cloud->point_step != aggregated_->point_step ||
               cloud->is_bigendian != aggregated_->is_bigendian ||
               !isFieldsEqual(cloud->fields, aggregated_->fields)) {
      RCLCPP_WARN(this->get_logger(), "Incoming cloud not compatible with aggregated layout (point_step/endianness/fields)");
      return;
    } else if (aggregated_->width >= static_cast<uint32_t>(max_points_)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Aggregated cloud reached max_points (%d). Ignoring further incoming clouds until reset.", max_points_);
      return;
    }

    // append data
    aggregated_->data.insert(aggregated_->data.end(), cloud->data.begin(), cloud->data.end());
    const size_t cloud_points = static_cast<size_t>(cloud->width) * static_cast<size_t>(cloud->height);
    aggregated_->width += static_cast<uint32_t>(cloud_points);
    aggregated_->height = 1;
    aggregated_->row_step = aggregated_->width * aggregated_->point_step;
    aggregated_->header.frame_id = fixed_frame_;
    ++aggregated_frames_;
  }

  void downsample_aggregated()
  {
    std::lock_guard<std::mutex> lock(agg_mutex_);
    // RCLCPP_INFO(this->get_logger(), "Downsampling aggregated cloud with %u points", aggregated_ ? aggregated_->width : 0);
    if (!aggregated_ || aggregated_->width == 0) {
      return;
    }
    try {
      pcl::PCLPointCloud2::Ptr input(new pcl::PCLPointCloud2(*aggregated_));
      pcl::PCLPointCloud2::Ptr output(new pcl::PCLPointCloud2());
      pcl::VoxelGrid<pcl::PCLPointCloud2> vox;
      vox.setInputCloud(input);
      vox.setLeafSize(static_cast<float>(voxel_leaf_size_),
                      static_cast<float>(voxel_leaf_size_),
                      static_cast<float>(voxel_leaf_size_));
      vox.filter(*output);

      // Preserve header/frame
      output->header.frame_id = fixed_frame_;
      output->header.stamp = input->header.stamp;
      aggregated_ = output;
      RCLCPP_INFO(this->get_logger(), "Downsampled aggregated cloud to width=%u", aggregated_->width);
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Downsampling failed: %s", e.what());
    }
  }

  void publish_map()
  {
    try {
      {
        std::lock_guard<std::mutex> lock(agg_mutex_);
        if (!aggregated_ || aggregated_->width == 0) return;
      }
  
      filter_aggregated_in_robot_frame();
      downsample_aggregated();
      std::lock_guard<std::mutex> lock(agg_mutex_);
      // Convert PCL aggregated cloud back to sensor_msgs for publishing
      sensor_msgs::msg::PointCloud2 out_msg;
      pcl::PCLPointCloud2 pcl_out = *aggregated_;
      // Update stamp to current time
      pcl_out.header.stamp = static_cast<std::uint64_t>(this->now().nanoseconds());
      pcl_out.header.frame_id = fixed_frame_;
      pcl_conversions::fromPCL(pcl_out, out_msg);
      out_msg.header.stamp = this->now();
      out_msg.header.frame_id = fixed_frame_;
      publisher_->publish(out_msg);
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Publish failed: %s", e.what());
    }
  }

  void handle_reset(const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/,
                    std::shared_ptr<std_srvs::srv::Empty::Response> /*res*/)
  {
    std::lock_guard<std::mutex> lock(agg_mutex_);
    aggregated_.reset();
    received_count_ = 0;
    RCLCPP_INFO(this->get_logger(), "Map reset: cleared aggregated cloud and stored clouds.");
  }

  // parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string fixed_frame_;
  std::string robot_frame_;
  double publish_period_;
  bool enable_transform_;
  double lookup_timeout_;
  int log_every_n_;
  int received_count_{0};
  bool acc_cloud_registered_;
  int aggregated_frames_{0};
  double voxel_leaf_size_;
  double pass_x_min_;
  double pass_x_max_;
  double pass_y_min_;
  double pass_y_max_;
  double pass_z_min_;
  double pass_z_max_;
  int max_points_;

  // tf
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

  // data
  pcl::PCLPointCloud2::Ptr aggregated_;
  std::mutex agg_mutex_;
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



