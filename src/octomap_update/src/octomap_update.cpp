#include <cstdio>
#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/srv/bounding_box_query.hpp>
#include <chrono>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rmw/qos_profiles.h>

// update octomap by registering a client to call the clear_bbox service

class OctomapUpdateNode : public rclcpp::Node
{
public:
  using BBoxSrv = octomap_msgs::srv::BoundingBoxQuery;

  OctomapUpdateNode()
  : Node("OctomapUpdateNode")
  {
    RCLCPP_INFO(this->get_logger(), "OctomapUpdateNode has been started.");
    this->declare_parameter<std::string>("input_topic", "/internimage/segmentation/ground_points/voxel");


    input_topic_ = this->get_parameter("input_topic").as_string();
    rclcpp::SensorDataQoS qos;
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&OctomapUpdateNode::update_octomap_callback, this, std::placeholders::_1));


    cli_group_cb = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    // Use default Services QoS (reliable) instead of SensorDataQoS (best-effort) to ensure matching
    service_client_ = this->create_client<BBoxSrv>("/octomap_server/clear_bbox", rclcpp::ServicesQoS(), cli_group_cb);

    
  }

  private:

  void handle_clear_bbox_response(rclcpp::Client<BBoxSrv>::SharedFuture future)
  {
    try {
      auto response = future.get();
      (void)response; // Not used currently
      RCLCPP_INFO(this->get_logger(), "Service call successful. Octomap BBox cleared.");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Service response exception: %s", e.what());
    }
  }

  void call_clear_bbox_service(
    double min_x, double min_y, double min_z,
    double max_x, double max_y, double max_z)
  {
    if (!service_client_->service_is_ready()) {
      // Non-blocking: avoid spinning a new executor inside a callback
      RCLCPP_WARN(this->get_logger(), "Service /octomap_server/clear_bbox not ready yet; skipping this bbox.");
      return;
    }

    auto request = std::make_shared<BBoxSrv::Request>();
    request->min.x = min_x;
    request->min.y = min_y;
    request->min.z = min_z;

    request->max.x = max_x;
    request->max.y = max_y;
    request->max.z = max_z;

    RCLCPP_INFO(this->get_logger(), "Sending request to clear BBox from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)",
      min_x, min_y, min_z, max_x, max_y, max_z);

    // send request
    auto result_future = service_client_->async_send_request(
      request,
      std::bind(&OctomapUpdateNode::handle_clear_bbox_response, this, std::placeholders::_1));
  }


  void update_octomap_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
    std::cout<<"update_octomap_callback"<<std::endl;
      bool has_x = false;
      bool has_y = false;
      bool has_z = false;
      for (const auto& field : msg->fields) {
          if (field.name == "x") has_x = true;
          if (field.name == "y") has_y = true;
          if (field.name == "z") has_z = true;
      }

      if (!has_x || !has_y || !has_z) {
          RCLCPP_WARN(rclcpp::get_logger("pointcloud_reader"), "Point cloud does not contain X, Y, Z fields.");
          return;
      }

      // Create iterators for x, y, z fields
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      // Iterate through the points and extract XYZ coordinates
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
          float x = *iter_x;
          float y = *iter_y;
          float z = *iter_z;

          float bbox_size = 0.05; 
          float min_x = x - bbox_size / 2;
          float min_y = y - bbox_size / 2;
          float min_z = z - bbox_size / 2;
          float max_x = x + bbox_size / 2;
          float max_y = y + bbox_size / 2;
          float max_z = z + bbox_size / 2;

          call_clear_bbox_service(min_x, min_y, min_z, max_x, max_y, max_z);

      }
    // for(size_t i=0; i < msg->data.size(); i += msg->point_step){
    //   float x = *reinterpret_cast<const float*>(&msg->data[i + msg->fields[0].offset]);
    //   float y = *reinterpret_cast<const float*>(&msg->data[i + msg->fields[1].offset]);
    //   float z = *reinterpret_cast<const float*>(&msg->data[i + msg->fields[2].offset]);
    // } 
    
  }

  std::string input_topic_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Client<BBoxSrv>::SharedPtr service_client_;
  rclcpp::CallbackGroup::SharedPtr cli_group_cb;

};



int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OctomapUpdateNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
