from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import launch_ros
from launch_ros.actions import Node, ComposableNodeContainer



def generate_launch_description():

    params_file = PathJoinSubstitution([
        FindPackageShare("octomap_update"), "config", "params.yaml"
    ])


    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")

    octomap_update_node = Node(
        package='octomap_update',
        executable='octomap_update',
        name='octomap_update',
        parameters=[params_file, {
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    

    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        octomap_update_node,
    ])
    return ld
        
    
