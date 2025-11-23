from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from ament_index_python.packages import get_package_share_directory
import os



def generate_launch_description():
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use simulation time if available.'
    )

    # internimage segmentation
    internimage_launch_file_path = os.path.join(
        get_package_share_directory('internimage'), 'launch', 'internimage_launch.py'
    )
    internimage_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(internimage_launch_file_path),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    )
    # depth image proc to pointcloud xyzrgb label   
    depth_proc_launch_file_path = os.path.join(
        get_package_share_directory('depth_image_proc'), 'launch', 'point_cloud_xyzrgb_label.launch.py'
    )
    depth_proc_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depth_proc_launch_file_path),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
            # namespace='my_namespace'
    )

    # depth cloud accumulation and voxel
    depth_cloud_acc_launch_file_path = os.path.join(
        get_package_share_directory('depth_cloud_acc'), 'launch', 'depth_cloud_acc.launch.py'
    )
    depth_cloud_acc_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depth_cloud_acc_launch_file_path),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
            # namespace='my_namespace'
    )


    # depth cloud acc to octomap
    depth_cloud_octomap_launch_file_path = os.path.join(
        get_package_share_directory('depth_cloud_acc'), 'launch', 'pc2octomap.launch.py'
    )
    depth_cloud_octomap_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depth_cloud_octomap_launch_file_path),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
            # namespace='my_namespace'
    )


    
    
    ld = LaunchDescription([
        use_sim_time_arg,
        internimage_node,
        depth_proc_node,
        depth_cloud_acc_node,
        depth_cloud_octomap_node,
    ])
    return ld
        
    
