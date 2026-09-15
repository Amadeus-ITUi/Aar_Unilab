##launch file
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 声明 launch 参数
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='hipnuc_config.yaml',
        description='配置文件名称（hipnuc_config.yaml 或 hipnuc_config_inference.yaml）'
    )
    
    # 获取配置文件路径（使用 PathJoinSubstitution 来处理 LaunchConfiguration）
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
                executable="imu_node",
                name="imu_node",
                parameters=[config],
                output="screen",
                # 可选：如果需要兼容原先 wit_ros2_imu 的 /imu/data 话题，可以取消注释
                # remappings=[
                #     ('/IMU_data', '/imu/data'),  # 将 /IMU_data remap 到 /imu/data
                # ],
            ),
        ]
    )
