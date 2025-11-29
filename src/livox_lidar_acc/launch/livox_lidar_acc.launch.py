from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("livox_lidar_acc"), "config", "params.yaml"
    ])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")


    container_pro_map = ComposableNodeContainer(
        name='container_pro_map',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # launch_ros.descriptions.ComposableNode(
            #     package='pcl_ros',
            #     plugin='pcl_ros::VoxelGrid',
            #     name='voxel_grid_node_acc_map',
            #     parameters=[
            #         {
            #             'use_sim_time': use_sim_time,
            #             'input_frame': 'world',
            #             'output_frame': 'world',  
            #             'leaf_size': 0.1,
            #             'filter_field_name': 'z',
            #             'filter_limit_min': -1000.0,
            #             'filter_limit_max': 1000.0,
            #         }
            #     ],
            #     remappings=[('input', '/internimage/segmentation/acc_global_map'),
            #                 ('output', '/internimage/segmentation/acc_global_map_voxelized')]
            # ),

            launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::PassThrough',
                    name='livox_points_passthrough_x_filter_node',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'input_frame': 'aliengo',
                        'output_frame': 'aliengo',  
                        'filter_field_name': 'x',
                        'filter_limit_min': 0.5,
                        # with too many ground points, octomap server will be too slow and do not actually update
                        'filter_limit_max': 100.0,
                    }],
                    remappings=[('input', '/livox_points'),
                                ('output', '/livox_points/x_filtered')]
                ),

            launch_ros.descriptions.ComposableNode(
                package='pcl_ros',
                plugin='pcl_ros::PassThrough',
                name='livox_points_passthrough_z_filter_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'input_frame': 'aliengo',
                    'output_frame': 'world',  
                    'filter_field_name': 'z',
                    'filter_limit_min': -1.0,
                    # with too many ground points, octomap server will be too slow and do not actually update
                    'filter_limit_max': 2.0,
                }],
                remappings=[('input', '/livox_points/x_filtered'),
                            ('output', '/livox_points/x_filtered/z_filtered')]
            ),
        ],
        output='screen',
    )

    livox_lidar_acc_node = Node(
        package='livox_lidar_acc',
        executable='livox_lidar_acc',
        name='livox_lidar_acc',
        parameters=[params_file, {
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    

    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        container_pro_map,
        livox_lidar_acc_node,
    ])
    return ld
        
    
