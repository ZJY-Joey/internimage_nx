from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    octomap_points_filter = ComposableNodeContainer(
        name='octomap_points_filter',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[

            launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::PassThrough',
                    name='octomap_points_passthrough_z_filter_node',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'input_frame': 'aliengo',
                        'output_frame': 'world',  
                        'filter_field_name': 'z',
                        'filter_limit_min': -1.0,
                        'filter_limit_max': 0.5,
                    }],
                    remappings=[('input', '/internimage/segmentation/projected/points'),
                                ('output', '/internimage/segmentation/projected/points/filtered_z')]
                ),
        ],
        output='screen',
    )

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        parameters=[{
            'use_sim_time': use_sim_time,
            'frame_id': 'world',
            'resolution': 0.05,
            'sensor_model/max_range': 5.0,
            # 'filter_ground': True,
            # 'ground_filter/distance':0.2,
            # 'ground_filter/plane_distance' : -0.6,

        }], 
        remappings=[('cloud_in', '/internimage/segmentation/projected/points/filtered_z')],

    )
    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        octomap_points_filter,
        octomap_node,
    ])
    return ld
        
    
