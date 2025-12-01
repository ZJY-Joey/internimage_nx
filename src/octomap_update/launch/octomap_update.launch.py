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

    ground_container_pro_map = ComposableNodeContainer(
        name='ground_container_pro_map',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # launch_ros.descriptions.ComposableNode(
            #     package='pcl_ros',
            #     plugin='pcl_ros::VoxelGrid',
            #     name='ground_voxel_grid_node_acc_map',
            #     parameters=[
            #         {
            #             'use_sim_time': use_sim_time,
            #             'input_frame': 'aliengo',
            #             'output_frame': 'world',  
            #             'leaf_size': 0.1,
            #             'filter_field_name': 'z',
            #             'filter_limit_min': -1.0,
            #             'filter_limit_max': .0,
            #         }
            #     ],
            #     remappings=[('input', '/internimage/segmentation/ground_points'),
            #                 ('output', '/internimage/segmentation/ground_points')]
            # ),
        ],
        output='screen',
    )


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
        ground_container_pro_map,
        octomap_update_node,
    ])
    return ld
        
    
