from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("inference")

    # 键盘控制脚本路径（由 CMake 安装到 share/inference/scripts 下）
    keyboard_script = os.path.join(pkg_share, "scripts", "keyboard_cmd_vel.py")

    return LaunchDescription(
        [
            # 键盘控制节点（Python 脚本，使用 xterm 打开独立终端，避免 stdin 不是 TTY 的问题）
            ExecuteProcess(
                cmd=["xterm", "-T", "keyboard_cmd_vel", "-e", "/usr/bin/python3", keyboard_script],
                output="screen",
            ),
        ]
    )

