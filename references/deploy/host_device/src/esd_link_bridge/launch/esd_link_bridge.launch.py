from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory("esd_link_bridge")) / "config" / "esd_link_bridge.yaml"
    return LaunchDescription([
        Node(
            package="esd_link_bridge",
            executable="esd_link_bridge_node",
            name="esd_link_bridge_node",
            output="screen",
            parameters=[str(config)],
        )
    ])
