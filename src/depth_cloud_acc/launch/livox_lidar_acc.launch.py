from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("depth_cloud_acc"), "config", "livox_lidar_acc.yaml"
    ])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")


    container_livox_filter = ComposableNodeContainer(
        name='container_livox_filter',
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
                        'output_frame': 'world',  
                        'filter_field_name': 'x',
                        'filter_limit_min': 0.5,
                        'filter_limit_max': 100.0,
                    }],
                    remappings=[('input', '/livox_points'),
                                ('output', '/livox_points/filtered_x')]
                ),

            # for segmentation label project
            launch_ros.descriptions.ComposableNode(
                package='pcl_ros',
                plugin='pcl_ros::PassThrough',
                name='livox_points_passthrough_acc_x_filter_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'input_frame': 'aliengo',
                    'output_frame': 'aliengo',  
                    'filter_field_name': 'x',
                    'filter_limit_min': 0.0,
                    # with too many ground points, octomap server will be too slow and do not actually update
                    'filter_limit_max': 100.0,
                }],
                remappings=[('input', '/livox_points/filtered_x/acc'),
                            ('output', '/livox_points/filtered_x/acc/filtered_x')]
            ),
        ],
        output='screen',
    )

    livox_lidar_acc_node = Node(
        package='depth_cloud_acc',
        executable='depth_cloud_acc',
        name='depth_cloud_acc',
        parameters=[params_file, {
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    

    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        livox_lidar_acc_node,
        container_livox_filter,
    ])
    return ld
        
    
