##launch file for IMU debug mode
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 声明 launch 参数
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='hipnuc_config_debug.yaml',
        description='配置文件名称（默认为 hipnuc_config_debug.yaml）'
    )
    
    # 获取配置文件路径
    config = PathJoinSubstitution([
        FindPackageShare('hipnuc_imu'),
        'config',
        LaunchConfiguration('config_file'),
    ])

    return LaunchDescription(
        [
            config_file_arg,
            Node(
                package="hipnuc_imu",
                executable="imu_node_debug",
                name="imu_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )
