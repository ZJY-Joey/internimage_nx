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
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # default_rviz = os.path.join(get_package_share_directory('depth_image_proc'),
    #                             'launch', 'rviz/point_cloud_xyzrgb.rviz')

    use_sim_time_arg = DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use simulation time if available.'
        )

    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([

        use_sim_time_arg,

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
                        # 2天空, 3地板, 6道路, 9草地, 11人行道, 12人, 13地面泥土, 29田野, 46沙地, 52小径, 53楼梯, 54跑道, 59楼梯间, 61桥, 91土路，94土地，121台阶
                        # moutain 2, 3, 6, 11, 12, 52, 53, 54, 58, 59, 95, 121
                        # grassland 2, 3, 6, 9, 11, 12, 13, 29, 46, 52, 53, 54, 59, 61, 91, 94, 121
                        'use_sim_time': use_sim_time, # moutain   # yard 
                        'filter_labels': [12],   
                        'filter_keep': False,   # drop specified labels
                        'target_frame': 'rs_d455_color_optical_frame',
                        'outlier_reject_MeanK': 100 ,
                        'outlier_reject_StddevMulThresh': 0.1,
                    }],
                    # to be changed too many here
                    remappings=[('rgb/camera_info', '/camera/rs_d455/color/camera_info'),
                                ('combined/image_rect_combined', '/internimage/combined_segmentation_mask'),
                                ('depth_registered/image_rect','/camera/rs_d455/depth/image_rect_raw'), #   /zed/zed_node/depth/depth_registered
                                ('lidar/points', '/livox_points/x_filtered/acc/z_filtered'), #'/cloud_registered/filtered/acc/filtered'),
                                ('points', '/internimage/segmentation/projected/points'),
                                ('ground_points', '/internimage/segmentation/ground_points')]
                ),
                

                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::VoxelGrid',
                #     name='voxel_grid_node',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'aliengo',
                #         'output_frame': 'aliengo',  
                #         'leaf_size': 0.2,
                #         'filter_field_name': 'z',
                #         'filter_limit_min': -1000.0,
                #         'filter_limit_max': 1000.0,
                #         # 'min_points_per_voxel': 100,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/projected/points'),
                #                 ('output', '/internimage/segmentation/voxel/points')]
                # ),

                # for cloud acc
                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::PassThrough',
                #     name='voxeled_passthrough_filter_node',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'aliengo',
                #         'output_frame': 'aliengo',  
                #         'filter_field_name': 'z',
                #         'filter_limit_min': -0.5,
                #         'filter_limit_max': 100.0,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/projected/points'),
                #                 ('output', '/internimage/segmentation/filtered/points')]
                # ),

                # for octomap
                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::PassThrough',
                #     name='passthrough_filter_node',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'aliengo',
                #         'output_frame': 'aliengo',  
                #         'filter_field_name': 'z',
                #         'filter_limit_min': -1.0,
                #         'filter_limit_max': 2.0,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/projected/points'),
                #                 ('output', '/internimage/segmentation/filtered/points')]
                # ),

                # ground points filter for octomap  
                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::PassThrough',
                #     name='ground_passthrough_filter_node',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'aliengo',
                #         'output_frame': 'aliengo',  
                #         'filter_field_name': 'x',
                #         'filter_limit_min': 0.0,
                #         # with too many ground points, octomap server will be too slow and do not actually update
                #         'filter_limit_max': 5.0,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/ground_points'),
                #                 ('output', '/internimage/segmentation/ground_points/filtered')]
                # ),




                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::VoxelGrid',
                #     name='voxel_grid_node_ground',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'aliengo',
                #         'output_frame': 'world',  
                #         'leaf_size': 0.5,
                #         'filter_field_name': 'z',
                #         'filter_limit_min': -1000.0,
                #         'filter_limit_max': 1000.0,
                #         # 'min_points_per_voxel': 100,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/ground_points/filtered'),
                #                 ('output', '/internimage/segmentation/ground_points/filtered/voxel')]
                # ),
            ],
            output='screen',
        ),

        

        
        launch_ros.actions.Node(
            package='image_transport',
            executable='republish',
            name='confidence_decompress_node',
            output='screen',
            # match the CLI: ros2 run image_transport republish --ros-args -p in_transport:=compressedDepth -p out_transport:=raw \
            #   --remap in/compressedDepth:=/zed/.../compressedDepth --remap out:=/zed/.../decompressed \
            #   -p "ffmpeg_image_transport.decoders.hevc:=hevc_cuvid,hevc"
            parameters=[{
                'use_sim_time': use_sim_time,
                'in_transport': 'compressedDepth',
                'out_transport': 'raw',
                # keep the ffmpeg decoder preference as a comma-separated string (matches CLI usage)
                'ffmpeg_image_transport.decoders.hevc': 'hevc_cuvid,hevc',
            }],
            arguments=[
                'compressedDepth',
                'in:=/zed/zed_node/confidence/confidence_map/compressedDepth', # /zed/zed_node/depth/depth_registered/compressed  /zed/zed_node/depth/depth_registered/compressedDepth
                'raw',
                'out:=/zed/zed_node/confidence/confidence_map/decompressedConfidence',
            ],
            remappings=[
                ('in/compressedDepth', '/zed/zed_node/confidence/confidence_map/compressedDepth'),
                ('out/raw', '/zed/zed_node/confidence/confidence_map/decompressedConfidence'),
            ],
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
                'use_sim_time': use_sim_time,
                'in_transport': 'compressedDepth',
                'out_transport': 'raw',
                # keep the ffmpeg decoder preference as a comma-separated string (matches CLI usage)
                'ffmpeg_image_transport.decoders.hevc': 'hevc_cuvid,hevc',
            }],
            arguments=[
                'compressedDepth',
                'in:=/zed/zed_node/depth/depth_registered/compressedDepth', # /zed/zed_node/depth/depth_registered/compressed  /zed/zed_node/depth/depth_registered/compressedDepth
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
