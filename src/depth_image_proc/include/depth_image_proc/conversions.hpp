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
#include "image_geometry/pinhole_camera_model.h"

#include <opencv2/core/mat.hpp>

#include <depth_image_proc/depth_traits.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <iostream>

namespace depth_image_proc
{

// Handles float or uint16 depths
template<typename T>
void convertDepth(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
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
  const T * depth_row = reinterpret_cast<const T *>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  for (int v = 0; v < static_cast<int>(cloud_msg->height); ++v, depth_row += row_step) {
    for (int u = 0; u < static_cast<int>(cloud_msg->width); ++u, ++iter_x, ++iter_y, ++iter_z) {
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

// Handles float or uint16 depths
template<typename T>
void convertDepthRadial(
  const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg,
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
  cv::Mat & transform)
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
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg)
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
  sensor_msgs::msg::PointCloud2::SharedPtr & cloud_msg,
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
