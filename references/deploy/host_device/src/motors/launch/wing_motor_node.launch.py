from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("motors"),
        "config",
        "wing_motors.yaml",
    )
    return LaunchDescription(
        [
            Node(
                package="motors",
                executable="wing_motor_node",
                name="wing_motor_node",
                parameters=[config],
                output="screen",
            )
        ]
    )
