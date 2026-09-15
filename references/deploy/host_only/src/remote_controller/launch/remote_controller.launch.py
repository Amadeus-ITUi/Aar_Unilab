from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # 声明启动参数
        DeclareLaunchArgument(
            'command_rate',
            default_value='20.0',
            description='命令发布频率 (Hz)'
        ),
        DeclareLaunchArgument(
            'command_topic',
            default_value='/cmd_vel',
            description='命令话题名称'
        ),
        
        # 远程控制器节点
        Node(
            package='remote_controller',
            executable='remote_controller_node',
            name='remote_controller_node',
            output='screen',
            parameters=[{
                'command_rate': LaunchConfiguration('command_rate'),
                'command_topic': LaunchConfiguration('command_topic'),
            }],
        ),
    ])
