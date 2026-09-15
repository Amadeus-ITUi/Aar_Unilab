# Copyright 2026 lzh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'joy_dev',
            default_value='/dev/input/js0',
            description='Xbox controller device',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/cmd_vel',
            description='Velocity command topic consumed by inference',
        ),
        DeclareLaunchArgument(
            'max_linear_speed',
            default_value='1.0',
        ),
        DeclareLaunchArgument(
            'max_angular_speed',
            default_value='1.0',
        ),
        DeclareLaunchArgument(
            'max_linear_accel',
            default_value='3.0',
            description='Maximum linear.x command acceleration in m/s^2',
        ),
        DeclareLaunchArgument(
            'deadzone',
            default_value='0.05',
        ),
        DeclareLaunchArgument(
            'command_timeout',
            default_value='0.5',
            description='Publish zero velocity after Joy input is stale',
        ),
        # Joy node - publishes joy messages from Xbox controller
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{
                'dev': LaunchConfiguration('joy_dev'),
                'deadzone': LaunchConfiguration('deadzone'),
                'autorepeat_rate': 20.0,
            }],
        ),
        # Xbox velocity publisher
        Node(
            package='xbox',
            executable='xbox_vel_publisher',
            name='xbox_vel_publisher',
            output='screen',
            parameters=[{
                'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                'max_angular_speed': LaunchConfiguration('max_angular_speed'),
                'max_linear_accel': LaunchConfiguration('max_linear_accel'),
                'deadzone': LaunchConfiguration('deadzone'),
                'command_timeout': LaunchConfiguration('command_timeout'),
                'output_topic': LaunchConfiguration('output_topic'),
            }],
        ),
    ])
