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


#include <deque>
#include <memory>

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
    this->declare_parameter<double>("publish_period", 0.1);
    this->declare_parameter<int>("max_clouds", 100);
    this->declare_parameter<bool>("enable_transform", true);
    this->declare_parameter<double>("lookup_timeout", 0.5);
    this->declare_parameter<int>("max_aggregated_points", 1000000); // hard cap to prevent memory blowup
    this->declare_parameter<int>("rebuild_keep_last_clouds", 50); // when rebuilding, number of recent clouds to retain
    this->declare_parameter<std::string>("aggregation_strategy", "rebuild"); // or 'truncate'
    this->declare_parameter<int>("log_every_n", 20);
    this->declare_parameter<bool>("acc_cloud_registered", false);
    this->declare_parameter<int>("max_aggregated_frames", 10);
    this->declare_parameter<double>("voxel_leaf_size", 1.0);
    this->declare_parameter<int>("min_points_per_voxel", 5);

    input_topic_ = this->get_parameter("input_depth_points_topic").as_string();
    output_topic_ = this->get_parameter("output_depth_points_topic").as_string();
    fixed_frame_ = this->get_parameter("fixed_frame").as_string();
    publish_period_ = this->get_parameter("publish_period").as_double();
    max_clouds_ = this->get_parameter("max_clouds").as_int();
    enable_transform_ = this->get_parameter("enable_transform").as_bool();
    lookup_timeout_ = this->get_parameter("lookup_timeout").as_double();
    max_aggregated_points_ = this->get_parameter("max_aggregated_points").as_int();
    rebuild_keep_last_clouds_ = this->get_parameter("rebuild_keep_last_clouds").as_int();
    log_every_n_ = this->get_parameter("log_every_n").as_int();
    acc_cloud_registered_ = this->get_parameter("acc_cloud_registered").as_bool();
    max_aggregated_frames_ = this->get_parameter("max_aggregated_frames").as_int();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    min_points_per_voxel_ = this->get_parameter("min_points_per_voxel").as_int();

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

    RCLCPP_INFO(this->get_logger(), "DepthCloudAccNode started: sub='%s' pub='%s' frame='%s' period=%.3f max_agg=%d",
      input_topic_.c_str(), output_topic_.c_str(), fixed_frame_.c_str(), publish_period_, max_aggregated_points_);
  }

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // RCLCPP_INFO(this->get_logger(), "Received point cloud with %u points", msg->width);
    sensor_msgs::msg::PointCloud2::SharedPtr cloud = msg;
    if (enable_transform_ && msg->header.frame_id != fixed_frame_) {
      try {
        // RCLCPP_INFO(this->get_logger(), "msg-> header.stamp.sec: %u", msg->header.stamp.sec);
        geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
        fixed_frame_, msg->header.frame_id, rclcpp::Time(msg->header.stamp),
        rclcpp::Duration::from_seconds(0.2));
        // RCLCPP_INFO(this->get_logger(), "Transform found from %s to %s",
                // msg->header.frame_id.c_str(), fixed_frame_.c_str());
        sensor_msgs::msg::PointCloud2 transformed;
        // Use generic tf2::doTransform (PointCloud2 specialization provided by tf2_sensor_msgs header)
        tf2::doTransform(*msg, transformed, t);
        // RCLCPP_INFO(this->get_logger(), "PointCloud2 transformed to %s frame", fixed_frame_.c_str());
        transformed.header.frame_id = fixed_frame_;
        cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(transformed);
        RCLCPP_DEBUG(this->get_logger(), "Transformed cloud from %s to %s", msg->header.frame_id.c_str(), fixed_frame_.c_str());
      } catch (const std::exception & e) {
        RCLCPP_INFO(this->get_logger(), "TF lookup/transform failed: %s", e.what());
        // fall back to original
      }
    }
    // RCLCPP_INFO(this->get_logger(), "Adding cloud with %u points to aggregation", cloud->width);

    // append to deque and aggregated
    // Convert incoming cloud to PCLPointCloud2 for internal storage
    pcl::PCLPointCloud2::Ptr pcl_cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2 tmp;
    pcl_conversions::toPCL(*cloud, tmp);
    *pcl_cloud = tmp;
    pcl_cloud->header.frame_id = fixed_frame_;

    if (max_clouds_ > 0) {
      if ((int)clouds_.size() >= max_clouds_) {
        clouds_.pop_front();
      }
      clouds_.push_back(pcl_cloud);
    }
    append_to_aggregated(pcl_cloud);
    if (++received_count_ % std::max(1, log_every_n_) == 0) {
      // RCLCPP_INFO(this->get_logger(), "Received %d clouds | aggregated_points=%u", received_count_, aggregated_ ? aggregated_->width : 0);
    }
  }

  void append_to_aggregated(const pcl::PCLPointCloud2::Ptr & cloud)
  {
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
      for (const auto &field : aggregated_->fields) {
        RCLCPP_INFO(this->get_logger(), "Aggregated field: %s offset=%u datatype=%u count=%u",
          field.name.c_str(), field.offset, field.datatype, field.count);
      }
    } else if (cloud->point_step != aggregated_->point_step ||
               cloud->is_bigendian != aggregated_->is_bigendian ||
               !isFieldsEqual(cloud->fields, aggregated_->fields)) {
      RCLCPP_WARN(this->get_logger(), "Incoming cloud not compatible with aggregated layout (point_step/endianness/fields)");
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
    RCLCPP_INFO(this->get_logger(), "Downsampling aggregated cloud with %u points", aggregated_ ? aggregated_->width : 0);
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
      aggregated_ = output;
      RCLCPP_INFO(this->get_logger(), "Downsampled aggregated cloud to width=%u", aggregated_->width);
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "Downsampling failed: %s", e.what());
    }
  }

  void publish_map()
  {
    if (aggregated_) {
      downsample_aggregated();
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
    // if not cloud_regitered, enforce limit with aggregated points
    if(!acc_cloud_registered_ ) {
        if (!aggregated_ || max_aggregated_points_ <= 0) return;
        if (static_cast<int>(aggregated_->width) <= max_aggregated_points_) return;
      return;
    }
    // if cloud_registered, enforce limit with aggregated frames;
    else{
      if(!aggregated_ || max_aggregated_points_ <= 0) return; 
      if( aggregated_frames_ <= max_aggregated_frames_) return; 
    }


    // Default: rebuild from tail of deque
    if (clouds_.empty()) return; // should not happen
    auto rebuilt = std::make_shared<pcl::PCLPointCloud2>();
    bool base_set = false;
    int kept = 0;
    for (auto it = clouds_.rbegin(); it != clouds_.rend() && kept < rebuild_keep_last_clouds_; ++it) {
      const auto & c = *it;
      if (!base_set) {
        *rebuilt = *c;
        rebuilt->data = c->data;
        base_set = true;
      } else {
        if (c->point_step != rebuilt->point_step || c->is_bigendian != rebuilt->is_bigendian || !isFieldsEqual(c->fields, rebuilt->fields) || c->height != rebuilt->height) {
          // skip incompatible older cloud
          continue;
        }
        rebuilt->data.insert(rebuilt->data.end(), c->data.begin(), c->data.end());
        rebuilt->width += c->width;
        rebuilt->row_step = rebuilt->width * rebuilt->point_step;
      }
      ++kept;
      if(!acc_cloud_registered_){
        if (static_cast<int>(rebuilt->width) >= max_aggregated_points_) break; // reached cap
      }
      else{
        if(kept >= max_aggregated_frames_) break; // reached cap
      }
    }
    aggregated_ = rebuilt;
    // RCLCPP_WARN(this->get_logger(), "Rebuilt aggregated cloud from last %d clouds -> %u points (cap=%d)", kept, aggregated_->width, max_aggregated_points_);
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
  bool acc_cloud_registered_;
  int aggregated_frames_{0};
  int max_aggregated_frames_;
  double voxel_leaf_size_;
  int min_points_per_voxel_;



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
  std::deque<pcl::PCLPointCloud2::Ptr> clouds_;
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



