#include <cstdio>
#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/srv/bounding_box_query.hpp>
#include <chrono>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rmw/qos_profiles.h>
// TF2 includes for frame transformations
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <omp.h> 
#include "octomap_extra_msgs/srv/list_bounding_box_query.hpp"

// update octomap by registering a client to call the clear_bbox service  

class OctomapUpdateNode : public rclcpp::Node
{
public:
  using ListBBoxSrv = octomap_extra_msgs::srv::ListBoundingBoxQuery;

  OctomapUpdateNode()
  : Node("OctomapUpdateNode")
  {
    RCLCPP_INFO(this->get_logger(), "OctomapUpdateNode has been started.");
    this->declare_parameter<std::string>("input_topic", "/internimage/segmentation/ground_points/voxel");
    this->declare_parameter<std::string>("target_frame", "map");


    input_topic_ = this->get_parameter("input_topic").as_string();
    rclcpp::SensorDataQoS qos;
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&OctomapUpdateNode::update_octomap_callback, this, std::placeholders::_1));


    cli_group_cb = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    // Use default Services QoS (reliable) instead of SensorDataQoS (best-effort) to ensure matching
    service_client_ = this->create_client<ListBBoxSrv>("/octomap_server/clear_list_bbox", rclcpp::ServicesQoS(), cli_group_cb);

    this->declare_parameter<float>("bbox_size", 0.10); 
    bbox_size_ = this->get_parameter("bbox_size").as_double();
    target_frame_ = this->get_parameter("target_frame").as_string();

    // TF Buffer & Listener (keep for lifetime of node)
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    
  }

  private:

  void handle_clear_bbox_response(rclcpp::Client<ListBBoxSrv>::SharedFuture future)
  {
    try {
      auto response = future.get();
      (void)response;
      // RCLCPP_INFO(this->get_logger(), "Service call successful. Octomap BBox cleared.");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Service response exception: %s", e.what());
    }
  }

  void call_clear_bbox_service(const std::vector<geometry_msgs::msg::Point> &ground_points, float bbox_size)
  {
    if (!service_client_->service_is_ready()) {
      RCLCPP_WARN(this->get_logger(), "Service /octomap_server/clear_bbox not ready yet; skipping this bbox.");
      return;
    }

    auto request = std::make_shared<ListBBoxSrv::Request>();
    request->center = ground_points;
    request->bbox_size = bbox_size;


    // RCLCPP_INFO(this->get_logger(), "Sending request to clear BBox from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)", min_x, min_y, min_z, max_x, max_y, max_z);

    // send request
    auto result_future = service_client_->async_send_request(
      request,
      std::bind(&OctomapUpdateNode::handle_clear_bbox_response, this, std::placeholders::_1));
  }


  void update_octomap_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){

    const std::string source_frame = msg->header.frame_id;
    if (source_frame.empty()) {
      RCLCPP_WARN(this->get_logger(), "PointCloud2 has empty frame_id; skip.");
      return;
    }

    // Check fields
    bool has_x=false, has_y=false, has_z=false;
    for (const auto & f : msg->fields){
      if (f.name=="x"){
        has_x=true; 
      } 
      if (f.name=="y") {
        has_y=true;
      } 
      if (f.name=="z"){
         has_z=true;
      }
    }
    if (!has_x || !has_y || !has_z){  
      RCLCPP_WARN(this->get_logger(), "PointCloud2 missing x/y/z fields; skip.");
      return;
    }

    if (!tf_buffer_->canTransform(target_frame_, source_frame, msg->header.stamp, tf2::durationFromSec(0.2))){
      RCLCPP_WARN(this->get_logger(), "Transform %s->%s not ready.", source_frame.c_str(), target_frame_.c_str());
      return;
    }

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    std::vector<geometry_msgs::msg::Point> ground_points;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){
      
      geometry_msgs::msg::PointStamped pt_src;
      pt_src.header.frame_id = source_frame;
      pt_src.header.stamp = msg->header.stamp;
      pt_src.point.x = *iter_x;
      pt_src.point.y = *iter_y;
      pt_src.point.z = *iter_z;
      geometry_msgs::msg::PointStamped pt_map;
      try {
        pt_map = tf_buffer_->transform(pt_src, target_frame_, tf2::durationFromSec(0.0));
      } catch (const tf2::TransformException & ex){
        continue; // skip this point
      }
      ground_points.push_back(pt_map.point);
      
     
    }
    double t1  = omp_get_wtime();
    call_clear_bbox_service(ground_points, this->bbox_size_);
    double t2 = omp_get_wtime();
    RCLCPP_INFO(this->get_logger(), "Time to clear bbox: %.6f seconds", t2 - t1);

  }

  float bbox_size_;
  std::string input_topic_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Client<ListBBoxSrv>::SharedPtr service_client_;
  rclcpp::CallbackGroup::SharedPtr cli_group_cb;
  std::string target_frame_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  

};



int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OctomapUpdateNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
