from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("livox_lidar_acc"), "config", "ground_points.yaml"
    ])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    ground_passthrough_container = ComposableNodeContainer(
        name='ground_points_passthrough_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # for cloud acc
            # in case some points in the back shouldn't be acc
                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::PassThrough',
                    name='ground_points_x_passthrough_x_node',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'input_frame': 'aliengo',
                        'output_frame': 'world',  
                        'filter_field_name': 'x',
                        'filter_limit_min': 0.0,
                        'filter_limit_max': 5.0,
                    }],
                    remappings=[('input', '/internimage/segmentation/ground_points/acc'),
                                ('output', '/internimage/segmentation/ground_points/acc/filtered_x')]
                ),

            
        ],
        output='screen',
    )


    ground_points_acc_node = Node(
        package='livox_lidar_acc',
        executable='livox_lidar_acc',
        name='livox_lidar_acc',
        parameters=[params_file, {'use_sim_time': use_sim_time,}],
        output='screen',
    )
    

    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        ground_points_acc_node,
        ground_passthrough_container,
    ])
    return ld
        
    
