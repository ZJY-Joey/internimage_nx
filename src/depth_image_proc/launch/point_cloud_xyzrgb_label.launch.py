# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# imports trimmed: get_package_share_directory and os were unused in this launch
from launch import LaunchDescription

import launch_ros.actions
import launch_ros.descriptions


def generate_launch_description():
    # default_rviz = os.path.join(get_package_share_directory('depth_image_proc'),
    #                             'launch', 'rviz/point_cloud_xyzrgb.rviz')
    return LaunchDescription([
        # install realsense from https://github.com/intel/ros2_intel_realsense
        # launch_ros.actions.Node(
        #     package='realsense_ros2_camera', executable='realsense_ros2_camera',
        #     output='screen'),

        # launch plugin through rclcpp_components container
        launch_ros.actions.ComposableNodeContainer(
            name='container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                # Driver itself
                launch_ros.descriptions.ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::PointCloudXyzrgbLabelNode',
                    name='point_cloud_xyzrgb_label_node',
                    parameters=[{
                        'use_sim_time': True,
                        'filter_labels': [3, 6, 9, 11, 30, 52, 53, 54, 58, 59, 95, 120],  
                        'filter_keep': False,   # drop specified labels
                    }],
                    remappings=[('rgb/camera_info', '/zed/zed_node/rgb/color/rect/camera_info'),
                                ('rgb/image_rect_color', '/internimage/color_segmentation_mask'),
                                ('id/image_rect_id', '/internimage/id_segmentation_mask'),
                                ('depth_registered/image_rect','/zed/zed_node/depth/depth_registered/decompressed'), #   /zed/zed_node/depth/depth_registered
                                ('points', '/internimage/segmentation/projected/points')]
                ),
                

                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::VoxelGrid',
                    name='voxel_grid_node',
                    parameters=[{
                        'use_sim_time': True,
                        'input_frame': 'aliengo',
                        'leaf_size': 0.05,
                        'filter_field_name': 'z',
                        'filter_limit_min': -1000.0,
                        'filter_limit_max': 1000.0,
                        # 'min_points_per_voxel': 100,
                    }],
                    remappings=[('input', '/internimage/segmentation/projected/points'),
                                ('output', '/internimage/segmentation/voxel/points')]
                ),

                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::PassThrough',
                    name='passthrough_filter_node',
                    parameters=[{
                        'user_sim_time': True,
                        'input_frame': 'aliengo',
                        'filter_field_name': 'z',
                        'filter_limit_min': -1.0,
                        'filter_limit_max': 0.5,
                    }],
                    remappings=[('input', '/internimage/segmentation/voxel/points'),
                                ('output', '/internimage/segmentation/filtered/points')]
                ),
            ],
            output='screen',
        ),

        

        # depth image decompressor
        launch_ros.actions.Node(
            package='image_transport',
            executable='republish',
            name='depth_decompress_node',
            output='screen',
            # match the CLI: ros2 run image_transport republish --ros-args -p in_transport:=compressedDepth -p out_transport:=raw \
            #   --remap in/compressedDepth:=/zed/.../compressedDepth --remap out:=/zed/.../decompressed \
            #   -p "ffmpeg_image_transport.decoders.hevc:=hevc_cuvid,hevc"
            parameters=[{
                'in_transport': 'compressedDepth',
                'out_transport': 'raw',
                # keep the ffmpeg decoder preference as a comma-separated string (matches CLI usage)
                'ffmpeg_image_transport.decoders.hevc': 'hevc_cuvid,hevc',
            }],
            arguments=[
                'compressedDepth',
                'in:=/zed/zed_node/depth/depth_registered/compressedDepth',
                'raw',
                'out:=/zed/zed_node/depth/depth_registered/decompressed',
            ],
            remappings=[
                ('in/compressedDepth', '/zed/zed_node/depth/depth_registered/compressedDepth'),
                ('out/raw', '/zed/zed_node/depth/depth_registered/decompressed'),
            ],
        ),

        # rviz
        # launch_ros.actions.Node(
        #     package='rviz2', executable='rviz2', output='screen',
        #     arguments=['--display-config', default_rviz]),
    ])
