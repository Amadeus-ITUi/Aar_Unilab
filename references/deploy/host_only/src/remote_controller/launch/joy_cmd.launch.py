from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 手柄设备参数（传给 joy_node）
        DeclareLaunchArgument(
            'joy_dev',
            default_value='/dev/input/js0',
            description='手柄设备路径，传给 joy_node 的 joy_dev 参数',
        ),
        DeclareLaunchArgument(
            'deadzone',
            default_value='0.1',
            description='摇杆死区，传给 joy_node',
        ),
        # 发布到推理节点的 /cmd_vel 参数
        DeclareLaunchArgument(
            'cmd_topic',
            default_value='/cmd_vel',
            description='手柄高层命令发布的话题（InferenceNode 订阅的 /cmd_vel）',
        ),
        DeclareLaunchArgument(
            'max_vx',
            default_value='1.5',
            description='最大前向线速度 (m/s)，对应 joy_cmd_node 的 max_vx',
        ),
        DeclareLaunchArgument(
            'max_dyaw',
            default_value='1.0',
            description='最大转向角速度 (rad/s)，对应 joy_cmd_node 的 max_dyaw',
        ),
        DeclareLaunchArgument(
            'max_height',
            default_value='0.3',
            description='最大高度命令幅值，对应 joy_cmd_node 的 max_height',
        ),
        DeclareLaunchArgument(
            'height_step',
            default_value='0.02',
            description='每次按键改变的高度增量，对应 joy_cmd_node 的 height_step',
        ),

        # 1）底层手柄驱动节点（标准 ROS2 joy）
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{
                'dev':        LaunchConfiguration('joy_dev'),
                'deadzone':   LaunchConfiguration('deadzone'),
            }],
        ),

        # 2）手柄 -> 推理命令节点（本包的 joy_cmd_node）
        Node(
            package='remote_controller',
            executable='joy_cmd_node',
            name='joy_cmd_node',
            output='screen',
            parameters=[{
                'cmd_topic':   LaunchConfiguration('cmd_topic'),
                'max_vx':      LaunchConfiguration('max_vx'),
                'max_dyaw':    LaunchConfiguration('max_dyaw'),
                'max_height':  LaunchConfiguration('max_height'),
                'height_step': LaunchConfiguration('height_step'),
                'deadzone':    LaunchConfiguration('deadzone'),
            }],
        ),
    ])

