## DR002 轮腿 Isaac Lab 策略推理节点 launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("inference")

    config = (os.path.join(pkg_share, "config", "lab_inference.yaml"),)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "joy_policy_gate_enabled",
                default_value="false",
                description="Use /joy buttons to gate policy output. Standby publishes zero commands.",
            ),
            DeclareLaunchArgument(
                "joy_policy_start_enabled",
                default_value="false",
                description="Initial policy gate state when joy_policy_gate_enabled is true.",
            ),
            DeclareLaunchArgument(
                "joy_policy_enable_button",
                default_value="2",
                description="Joy button index that switches to policy mode. Xbox default: X=2.",
            ),
            DeclareLaunchArgument(
                "force_zero_policy_commands",
                default_value="false",
                description=(
                    "Publish native six-axis zero commands while POLICY authority is enabled."
                ),
            ),
            DeclareLaunchArgument(
                "joy_policy_standby_button",
                default_value="3",
                description="Joy button index that switches to standby/def-pos mode. Xbox default: Y=3.",
            ),
            Node(
                package="inference",
                executable="lab_inference_node",
                name="lab_inference_node",
                parameters=[
                    config,
                    {
                        "joy_policy_gate_enabled": ParameterValue(
                            LaunchConfiguration("joy_policy_gate_enabled"), value_type=bool),
                        "joy_policy_start_enabled": ParameterValue(
                            LaunchConfiguration("joy_policy_start_enabled"), value_type=bool),
                        "joy_policy_enable_button": ParameterValue(
                            LaunchConfiguration("joy_policy_enable_button"), value_type=int),
                        "joy_policy_standby_button": ParameterValue(
                            LaunchConfiguration("joy_policy_standby_button"), value_type=int),
                        "force_zero_policy_commands": ParameterValue(
                            LaunchConfiguration("force_zero_policy_commands"), value_type=bool),
                    },
                ],
                output="screen",
            ),
        ]
    )
