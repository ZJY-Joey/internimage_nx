from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # image_topic_arg = DeclareLaunchArgument(
    #     "image_topic",
    #     default_value="",
    #     description="Image topic to subscribe (overrides YAML)",
    # )

    use_sim_time_arg = DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use simulation time if available.'
        )

    # image_topic = LaunchConfiguration("image_topic")
    use_sim_time = LaunchConfiguration("use_sim_time")

    params_file = PathJoinSubstitution(
        [FindPackageShare("internimage"), "config", "params.yaml"]
    )

    internimage_node = Node(
            package='internimage',
            executable='internimage',
            name='internimage',
            output='screen',
            parameters=[params_file,{'use_sim_time': use_sim_time}],
        )
    
    ld = LaunchDescription()
    # ld.add_action(image_topic_arg)
    ld.add_action(use_sim_time_arg)
    ld.add_action(internimage_node)

    return ld
        
    
