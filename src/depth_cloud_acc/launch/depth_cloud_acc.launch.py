from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("depth_cloud_acc"), "config", "params.yaml"
    ])

    # Launch arguments
    publish_period_arg = DeclareLaunchArgument(
        'publish_period', default_value='0.2',
        description='Publish period (seconds) for aggregated map.'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use simulation time if available.'
    )


    container_pro_map = ComposableNodeContainer(
        name='container_pro_map',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            launch_ros.descriptions.ComposableNode(
                package='pcl_ros',
                plugin='pcl_ros::VoxelGrid',
                name='voxel_grid_node',
                parameters=[params_file],
                remappings=[('input', '/internimage/segmentation/acc_global_map'),
                            ('output', '/internimage/segmentation/acc_global_map_voxelized')]
            ),
        ],
        output='screen',
    )

    depth_cloud_acc_node = Node(
        package='depth_cloud_acc',
        executable='depth_cloud_acc',
        name='depth_cloud_acc',
        parameters=[params_file, {
            'publish_period': LaunchConfiguration('publish_period'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen'
    )
    

    
    
    ld = LaunchDescription([
        publish_period_arg,
        use_sim_time_arg,
        container_pro_map,
        depth_cloud_acc_node,
    ])
    return ld
        
    
