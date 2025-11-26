from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("depth_cloud_acc"), "config", "cloud_registered.yaml"
    ])

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    cloud_registered_passthrough_container = ComposableNodeContainer(
        name='cloud_registered_passthrough_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # for cloud acc
                launch_ros.descriptions.ComposableNode(
                    package='pcl_ros',
                    plugin='pcl_ros::PassThrough',
                    name='cloud_registered_passthrough_x_node',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'input_frame': 'aliengo',
                        'output_frame': 'camera_init',  
                        'filter_field_name': 'x',
                        'filter_limit_min': 0.0,
                        'filter_limit_max': 100.0,
                    }],
                    remappings=[('input', '/cloud_registered/acc'),
                                ('output', '/cloud_registered/acc/filtered')]
                ),

                # launch_ros.descriptions.ComposableNode(
                #     package='pcl_ros',
                #     plugin='pcl_ros::PassThrough',
                #     name='cloud_registered_passthrough_x_node',
                #     parameters=[{
                #         'use_sim_time': use_sim_time,
                #         'input_frame': 'camera_init',
                #         'output_frame': 'camera_init',  
                #         'filter_field_name': 'x',
                #         'filter_limit_min': -1.0,
                #         'filter_limit_max': 0.5,
                #     }],
                #     remappings=[('input', '/internimage/segmentation/voxel/points'),
                #                 ('output', '/internimage/segmentation/voxel/filtered/points')]
                # ),
            
        ],
        output='screen',
    )


    depth_cloud_acc_node = Node(
        package='depth_cloud_acc',
        executable='depth_cloud_acc',
        name='depth_cloud_acc',
        parameters=[params_file, {'use_sim_time': use_sim_time,}],
        output='screen'
    )
    

    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        depth_cloud_acc_node,
        cloud_registered_passthrough_container,
    ])
    return ld
        
    
