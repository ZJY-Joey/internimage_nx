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

    # image_topic = LaunchConfiguration("image_topic")
    # use_sim_time = LaunchConfiguration("use_sim_time")

    params_file = PathJoinSubstitution(
        [FindPackageShare("internimage"), "config", "params.yaml"]
    )
    # node_params = [params_file]

    internimage_node = Node(
            package='internimage',
            executable='internimage',
            name='internimage',
            output='screen',
            parameters=[params_file],
        )
    
    ld = LaunchDescription()
    # ld.add_action(image_topic_arg)
    # ld.add_action(use_sim_time)
    ld.add_action(internimage_node)

    return ld
        
    
