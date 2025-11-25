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

        }], 
        remappings=[('cloud_in', '/internimage/segmentation/voxel/filtered/points')],

    )
    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        octomap_node,
    ])
    return ld
        
    
