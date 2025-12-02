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
    rs_launch_file_path = os.path.join(
        get_package_share_directory('atec_internimage'), 'launch', 'rs_launch.py'
    )
    rs_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items()
    )

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

    livox_lidar_acc_launch_file_path = os.path.join(
        get_package_share_directory('livox_lidar_acc'), 'launch', 'livox_lidar_acc.launch.py'
    )
    livox_lidar_acc_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(livox_lidar_acc_launch_file_path),
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

    # global_map acc for sam3 
    global_map_launch_file_path = os.path.join(
        get_package_share_directory('depth_cloud_acc'), 'launch', 'global_map_for_sam3.launch.py'
    )
    global_map_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(global_map_launch_file_path),
        launch_arguments={'use_sim_time': use_sim_time }.items(),
    )

    # ground points for octomap clear
    ground_points_acc_launch_file_path = os.path.join(
        get_package_share_directory('livox_lidar_acc'), 'launch', 'ground_points_acc.launch.py'
    )
    ground_points_acc_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ground_points_acc_launch_file_path),
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
        # rs_node,
        # internimage_node,
        # cloud_registered_acc_node,
        livox_lidar_acc_node,
        # ground_points_acc_node,
        # depth_proc_node,
        # global_map_node,
        depth_cloud_octomap_node,
        # octomap_update_node,
    ])
    return ld
        
    
