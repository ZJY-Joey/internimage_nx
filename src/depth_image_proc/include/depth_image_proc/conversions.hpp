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

#ifndef DEPTH_IMAGE_PROC__CONVERSIONS_HPP_
#define DEPTH_IMAGE_PROC__CONVERSIONS_HPP_

#include <limits>
#include <unordered_set>
#include <vector>
#include <cmath>

#include "image_geometry/pinhole_camera_model.hpp"

#include <opencv2/core/mat.hpp>

#include <depth_image_proc/depth_traits.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <iostream>
#include <fstream>

namespace depth_image_proc
{

// Handles float or uint16 depths
template<typename T>
void convertDepth(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const image_geometry::PinholeCameraModel & model,
  double invalid_depth = 0.0)
{
  // Use correct principal point from calibration
  float center_x = model.cx();
  float center_y = model.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = DepthTraits<T>::toMeters(T(1));
  float constant_x = unit_scaling / model.fx();
  float constant_y = unit_scaling / model.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  // ensure that the computation only happens in case we have a default depth
  T invalid_depth_cvt = T(0);
  bool use_invalid_depth = invalid_depth != 0.0;
  if (use_invalid_depth) {
    invalid_depth_cvt = DepthTraits<T>::fromMeters(invalid_depth);
  }
  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");

  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  uint32_t row_step = depth_msg->step / sizeof(T);
  for (uint32_t v = 0; v < cloud_msg->height; ++v, depth_row += row_step) {
    for (uint32_t u = 0; u < cloud_msg->width; ++u, ++iter_x, ++iter_y, ++iter_z) {
      T depth = depth_row[u];

      // Missing points denoted by NaNs
      if (!DepthTraits<T>::valid(depth)) {
        if (use_invalid_depth) {
          depth = invalid_depth_cvt;
        } else {
          *iter_x = *iter_y = *iter_z = bad_point;
          continue;
        }
      }

      // Fill in XYZ
      *iter_x = (u - center_x) * depth * constant_x;
      *iter_y = (v - center_y) * depth * constant_y;
      *iter_z = DepthTraits<T>::toMeters(depth);
    }
  }
}

// Handles float or uint16 depths
template<typename T>
void convertDepthwithLabel(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & id_msg,
  std::unordered_set<unsigned char> filter_labels,
  bool filter_keep,
  const image_geometry::PinholeCameraModel & model,
  double range_max = 0.0)
{
  // Use correct principal point from calibration
  float center_x = model.cx();
  float center_y = model.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = DepthTraits<T>::toMeters(T(1) );
  float constant_x = unit_scaling / model.fx();
  float constant_y = unit_scaling / model.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");  
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_label(*cloud_msg, "label");
  const uint8_t * id_ptr = &id_msg->data[0];

  // Best-effort bytes-per-pixel for id image (supports 8U/16U/32S typical encodings)
  int id_pixel_step = static_cast<int>(id_msg->step / id_msg->width);
  if (id_pixel_step <= 0) {
    id_pixel_step = 1;  // fallback for safety
  }
  int id_skip = static_cast<int>(id_msg->step - id_msg->width * id_pixel_step);

  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += row_step, id_ptr += id_skip) {
    for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_x, ++iter_y, ++iter_z, id_ptr += id_pixel_step, ++iter_label) {
      T depth = depth_row[u];

      // Missing points denoted by NaNs
      if (!DepthTraits<T>::valid(depth)) {
        if (range_max != 0.0) {
          depth = DepthTraits<T>::fromMeters(range_max);
        } else {
          *iter_x = *iter_y = *iter_z = bad_point;
          continue;
        }
      }

      // Fill in XYZ
      *iter_x = (u - center_x) * depth * constant_x;
      *iter_y = (v - center_y) * depth * constant_y;
      *iter_z = DepthTraits<T>::toMeters(depth);
      
      // Semantic label
      uint8_t label_value = 0;
      if (id_pixel_step == 1) {
        label_value = id_ptr[0];
      } else if (id_pixel_step == 2) {
        uint16_t v16 = 0;
        std::memcpy(&v16, id_ptr, sizeof(uint16_t));
        label_value = static_cast<uint8_t>(v16);
      } else if (id_pixel_step >= 4) {
        // Covers 32-bit integer labels; truncate to 8-bit as classes < 155
        uint32_t v32 = 0;
        std::memcpy(&v32, id_ptr, sizeof(uint32_t));
        label_value = static_cast<uint8_t>(v32);
      }
      *iter_label = label_value;
      // std::cout<<"label value: "<<static_cast<int>(label_value)<<std::endl;
      const bool in_set = (filter_labels.find(label_value) != filter_labels.end());
      const bool should_mask = filter_keep ? !in_set : in_set;
      if (should_mask) {
        *iter_x = bad_point;
        *iter_y = bad_point;
        *iter_z = bad_point;
      }

    }
  }
}



template<typename T>
void convertDepthwithCombinedmsg(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & combined_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & conf_msg,
  std::unordered_set<unsigned char> filter_labels,
  bool filter_keep,
  const image_geometry::PinholeCameraModel & model,
  int red_offset, int green_offset, int blue_offset, int color_step,
  double range_max = 0.0)
  {

  // std::cout<<"call convertDepthwithCombinedmsg"<<std::endl;
  // Use correct principal point from calibration
  float center_x = model.cx();
  float center_y = model.cy();

  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  double unit_scaling = DepthTraits<T>::toMeters(T(1) );
  float constant_x = unit_scaling / model.fx();
  float constant_y = unit_scaling / model.fy();
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");  
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_label(*cloud_msg, "label");
  const uint8_t * combined_ptr = &combined_msg->data[0];

  // Best-effort bytes-per-pixel for combined image (supports 8U/16U/32S typical encodings)
  int combined_pixel_step = static_cast<int>(combined_msg->step / combined_msg->width);
  if (combined_pixel_step <= 0) {
    combined_pixel_step = 1;  // fallback for safety
  }
  int combined_skip = static_cast<int>(combined_msg->step - combined_msg->width * combined_pixel_step);

  const uint8_t * conf_ptr = &conf_msg->data[0];

  // Best-effort bytes-per-pixel for id image (supports 8U/16U/32S typical encodings)
  int conf_pixel_step = static_cast<int>(conf_msg->step / conf_msg->width);
  if (conf_pixel_step <= 0) {
    conf_pixel_step = 1;  // fallback for safety
  }
  int conf_skip = static_cast<int>(conf_msg->step - conf_msg->width * conf_pixel_step);

  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  int count = 0;
  // Collect unique labels seen in this frame so we can write them once per frame
  std::unordered_set<int> seen_labels;
  // float max_conf = 0.0;
  for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += row_step, combined_ptr += combined_skip, conf_ptr += conf_skip) {
    for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b, combined_ptr += combined_pixel_step, ++iter_label, conf_ptr += conf_pixel_step) {
      T depth = depth_row[u];

      // Missing points denoted by NaNs
      if (!DepthTraits<T>::valid(depth)) {
        if (range_max != 0.0) {
          depth = DepthTraits<T>::fromMeters(range_max);
        } else {
          *iter_x = *iter_y = *iter_z = bad_point;
          continue;
        }
      }

      // Fill in XYZ
      *iter_x = (u - center_x) * depth * constant_x;
      *iter_y = (v - center_y) * depth * constant_y;
      *iter_z = DepthTraits<T>::toMeters(depth);

      // std::cout<<"x: "<<*iter_x<<" y: "<<*iter_y<<" z: "<<*iter_z<<std::endl;

      
      // Semantic label: for RGBA8, the 4th channel (A) carries the label per pixel.
      // Validate that we indeed have 4 bytes per pixel.
      if (combined_pixel_step != 4) {
        throw std::runtime_error("convertDepthwithCombinedmsg expects RGBA8 (4 bytes per pixel) for combined_msg");
      }
      const int label_offset = 3;  // RGBA8 -> A channel index
      *iter_label = combined_ptr[label_offset];
      // Track the label for a per-frame summary
      seen_labels.insert(static_cast<int>(*iter_label));
      const bool in_set = (filter_labels.find(*iter_label) != filter_labels.end());
      const bool should_mask = filter_keep ? !in_set : in_set;
      if (should_mask) {
        *iter_x = bad_point;
        *iter_y = bad_point;
        *iter_z = bad_point;
      }

      // rgb values
      *iter_r = combined_ptr[red_offset];
      *iter_g = combined_ptr[green_offset];
      *iter_b = combined_ptr[blue_offset];

      // std::cout<<"r: "<<static_cast<int>(*iter_r)<<" g: "<<static_cast<int>(*iter_g)<<" b: "<<static_cast<int>(*iter_b)<<std::endl;

      // Confidence value filter
      float conf_value = 0;
      if (conf_pixel_step == 1) {
        conf_value = conf_ptr[0];
      } else if (conf_pixel_step == 2) {
        uint16_t v16 = 0;
        std::memcpy(&v16, conf_ptr, sizeof(uint16_t));
        conf_value = static_cast<float>(v16);
      } else if (conf_pixel_step == 4) {
        // Covers 32-bit integer labels; truncate to 8-bit as classes < 155
        float v32 = 0;
        std::memcpy(&v32, conf_ptr, sizeof(float));
        conf_value = static_cast<float>(v32);
      } else {
        throw std::runtime_error("Unsupported confidence image encoding");
      }
      // std::cout<<"conf value: "<<static_cast<int>(conf_value)<<std::endl;
      // conf 200 for task3   conf20 for task1 and task2
      // if (conf_value > max_conf) {
      //   max_conf = conf_value;
      // }
      if (conf_value > 20) {  // threshold can be parameterized
        count++;
        *iter_x = bad_point;
        *iter_y = bad_point;
        *iter_z = bad_point;
      }

      
      

    }
  }
  // std::cout<<"max conf value in frame: "<<max_conf<<std::endl;
  // complie if in need
    // char filename[ ] = "label_value.txt";
    // std::fstream myFile(filename, std::fstream::out | std::fstream::app);
    // // After processing the whole frame, write a single summary line listing unique labels
    // if (myFile.is_open() || true) {
    //   // Re-open for append if it was closed above (some platforms may have closed it earlier)
    //   if (!myFile.is_open()) myFile.open(filename, std::fstream::out | std::fstream::app);
    //   if (myFile.is_open()) {
    //     // Use depth_msg timestamp if available
    //     uint32_t sec = depth_msg->header.stamp.sec;
    //     uint32_t nsec = depth_msg->header.stamp.nanosec;
    //     myFile << sec << '.' << nsec << ' ';
    //     bool first = true;
    //     for (int lbl : seen_labels) {
    //       if (!first) myFile << ',';
    //       first = false;
    //       const bool in_set = (filter_labels.find(static_cast<unsigned char>(lbl)) != filter_labels.end());
    //       myFile << lbl << ':' << (in_set ? "in" : "out");
    //     }
    //     myFile << '\n';
    //     myFile.close();
    //   }
    // }
    // std::cout<<"conf filter count: "<<count<<std::endl;

  }




template<typename T>
void convertLabelAndRgbWithLidar(
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & combined_msg,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & lidar_msg,
  std::unordered_set<unsigned char> filter_labels,
  bool filter_keep,
  const image_geometry::PinholeCameraModel & model,
  int red_offset, int green_offset, int blue_offset)
{
  // std::cout<<"call convertLabelAndRgbWithLidar"<<std::endl;
  // If no LiDAR, output empty cloud
  if (!lidar_msg) {
    cloud_msg->data.clear();
    cloud_msg->width = 0;
    cloud_msg->height = 1;
    cloud_msg->row_step = 0;
    cloud_msg->is_dense = false;
    // std::cout << "cloud_msg points: 0" << std::endl;
    return;
  }

  const int img_w = static_cast<int>(combined_msg->width);
  const int img_h = static_cast<int>(combined_msg->height);
  const int combined_pixel_step = static_cast<int>(combined_msg->step / combined_msg->width);
  if (combined_pixel_step != 4) {
    throw std::runtime_error("convertLabelAndRgbWithLidar expects RGBA8 (4 bytes per pixel) for combined_msg");
  }


  const size_t point_step = cloud_msg->point_step;
  int offset_x = -1, offset_y = -1, offset_z = -1, offset_rgb = -1, offset_label = -1;
  for (const auto & f : cloud_msg->fields) {
    if (f.name == "x") offset_x = f.offset;
    else if (f.name == "y") offset_y = f.offset;
    else if (f.name == "z") offset_z = f.offset;
    else if (f.name == "rgb") offset_rgb = f.offset;
    else if (f.name == "label") offset_label = f.offset;
  }
  if (offset_x < 0 || offset_y < 0 || offset_z < 0 || offset_rgb < 0 || offset_label < 0) {
    throw std::runtime_error("PointCloud2 missing required fields (x,y,z,rgb,label)");
  }

  cloud_msg->data.clear();
  cloud_msg->data.shrink_to_fit();  
  cloud_msg->is_dense = false;    

  sensor_msgs::PointCloud2ConstIterator<float> in_x(*lidar_msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> in_y(*lidar_msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> in_z(*lidar_msg, "z");

  size_t written = 0;

  // Prepare a reusable buffer for one point record
  std::vector<uint8_t> one_point(point_step, 0);

  // define an depth image
  const size_t map_size = static_cast<size_t>(img_w * img_h);
  std::vector<float> min_distances(map_size, std::numeric_limits<float>::max());

  for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z) {
    const float X = *in_x;
    const float Y = *in_y;
    const float Z = *in_z;

    // Skip invalid or behind camera
    if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z) || Z <= 0.f) {
      continue;
    }

    float distance = std::sqrt(std::pow(X, 2) + std::pow(Y, 2) + std::pow(Z, 2));

    // Project LiDAR point to image
    const cv::Point2d uv = model.project3dToPixel(cv::Point3d(X, Y, Z));
    const int u = static_cast<int>(uv.x);
    const int v = static_cast<int>(uv.y);
    
    if (u < 0 || u >= img_w || v < 0 || v >= img_h) {
      continue; // discard if outside combined_msg bounds (no semantic label)
    }

    const size_t map_index = static_cast<size_t>(v * img_w + u);
    if (distance > min_distances[map_index]) {
      continue; 
    }
    min_distances[map_index] = distance;

    const int base = v * static_cast<int>(combined_msg->step) + u * combined_pixel_step;
    const uint8_t * px = &combined_msg->data[base];

    const uint8_t label = px[3];

    // Label filtering
    const bool in_set = (filter_labels.find(label) != filter_labels.end());
    const bool should_mask = filter_keep ? !in_set : in_set;
    if (should_mask) {
      continue; // discard
    }

    // Pack XYZ
    std::fill(one_point.begin(), one_point.end(), 0);
    std::memcpy(one_point.data() + offset_x, &X, sizeof(float));
    std::memcpy(one_point.data() + offset_y, &Y, sizeof(float));
    std::memcpy(one_point.data() + offset_z, &Z, sizeof(float));

    // Pack RGB from first three channels using provided offsets
    const uint8_t r = px[red_offset];
    const uint8_t g = px[green_offset];
    const uint8_t b = px[blue_offset];
    const uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                         (static_cast<uint32_t>(g) << 8)  |
                          static_cast<uint32_t>(b);
    std::memcpy(one_point.data() + offset_rgb, &rgb, sizeof(uint32_t));

    // Pack label
    std::memcpy(one_point.data() + offset_label, &label, sizeof(uint8_t));

    // Append to cloud buffer
    cloud_msg->data.insert(cloud_msg->data.end(), one_point.begin(), one_point.end());
    ++written;
  }

  // Finalize cloud shape
  cloud_msg->width = static_cast<uint32_t>(written);
  cloud_msg->height = 1;
  cloud_msg->row_step = static_cast<uint32_t>(point_step * written);
  cloud_msg->header.stamp = lidar_msg->header.stamp;

  // Output the number of points
  // std::cout << "cloud_msg points: " << written << std::endl;
}

// Handles float or uint16 depths
template<typename T>
void convertDepthRadial(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  const cv::Mat & transform)
{
  // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
  float bad_point = std::numeric_limits<float>::quiet_NaN();

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += row_step) {
    for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_x, ++iter_y, ++iter_z) {
      T depth = depth_row[u];

      // Missing points denoted by NaNs
      if (!DepthTraits<T>::valid(depth)) {
        *iter_x = *iter_y = *iter_z = bad_point;
        continue;
      }
      const cv::Vec3f & cvPoint = transform.at<cv::Vec3f>(u, v) * DepthTraits<T>::toMeters(depth);
      // Fill in XYZ
      *iter_x = cvPoint(0);
      *iter_y = cvPoint(1);
      *iter_z = cvPoint(2);
    }
  }
}

// Handles float or uint16 depths
template<typename T>
void convertIntensity(
  const sensor_msgs::msg::Image::ConstSharedPtr & intensity_msg,
  const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg)
{
  sensor_msgs::PointCloud2Iterator<float> iter_i(*cloud_msg, "intensity");
  const T * inten_row = reinterpret_cast<const T *>(&intensity_msg->data[0]);

  const int i_row_step = intensity_msg->step / sizeof(T);
  for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, inten_row += i_row_step) {
    for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_i) {
      *iter_i = inten_row[u];
    }
  }
}

// Handles RGB8, BGR8, and MONO8
void convertRgb(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
  const sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  int red_offset, int green_offset, int blue_offset, int color_step);

cv::Mat initMatrix(cv::Mat cameraMatrix, cv::Mat distCoeffs, int width, int height, bool radial);

void convertRgbLabel(
  const sensor_msgs::msg::Image::ConstSharedPtr & rgb_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr & id_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  int red_offset, int green_offset, int blue_offset, int color_step);


void convertLabel(
  const sensor_msgs::msg::Image::ConstSharedPtr & id_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg);

}  // namespace depth_image_proc

#endif  // DEPTH_IMAGE_PROC__CONVERSIONS_HPP_
