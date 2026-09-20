##launch file for sweep frequency node (motor testing)
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = (
        os.path.join(
            get_package_share_directory("inference"),
            "config",
            "sweep_frequency.yaml",
        ),
    )

    return LaunchDescription(
        [
            Node(
                package="inference",
                executable="sweep_frequency_node",
                name="sweep_frequency_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )

