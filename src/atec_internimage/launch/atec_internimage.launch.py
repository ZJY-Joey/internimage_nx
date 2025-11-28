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
        'use_sim_time', default_value='False',
        description='Use simulation time if available.'
    )

    use_sim_time = LaunchConfiguration("use_sim_time")


    # internimage segmentation
    internimage_launch_file_path = os.path.join(
        get_package_share_directory('internimage'), 'launch', 'internimage_launch.py'
    )
    internimage_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(internimage_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items()
    )

    # cloud_registered cloud accmulation
    cloud_registered_acc_launch_file_path = os.path.join(
        get_package_share_directory('depth_cloud_acc'), 'launch', 'cloud_registered_acc.launch.py'
    )
    cloud_registered_acc_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cloud_registered_acc_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )


    # depth image proc to pointcloud xyzrgb label   
    depth_proc_launch_file_path = os.path.join(
        get_package_share_directory('depth_image_proc'), 'launch', 'point_cloud_xyzrgb_label.launch.py'
    )
    depth_proc_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(depth_proc_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )

    # depth cloud accumulation and voxel
    depth_cloud_acc_launch_file_path = os.path.join(
        get_package_share_directory('depth_cloud_acc'), 'launch', 'depth_cloud_acc.launch.py'
    )
    depth_cloud_acc_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(depth_cloud_acc_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )


    # depth cloud acc to octomap
    depth_cloud_octomap_launch_file_path = os.path.join(
        get_package_share_directory('atec_internimage'), 'launch', 'pc2octomap.launch.py'
    )
    depth_cloud_octomap_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(depth_cloud_octomap_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )

    # octomap update clear bbox node
    octomap_update_launch_file_path = os.path.join(
        get_package_share_directory('octomap_update'), 'launch', 'octomap_update.launch.py'
    )
    octomap_update_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(octomap_update_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )


    ld = LaunchDescription([
        use_sim_time_arg,
        internimage_node,
        cloud_registered_acc_node,
        depth_proc_node,
        # depth_cloud_acc_node,
        depth_cloud_octomap_node,
        octomap_update_node,
    ])
    return ld
        
    
