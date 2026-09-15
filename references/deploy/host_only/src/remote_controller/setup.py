from setuptools import setup
import os
from glob import glob

package_name = 'remote_controller'

setup(
    name=package_name,
    version='0.0.0',
    # 源码在 src/remote_controller 下
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='远程控制器节点 - 使用Xbox手柄控制机器人',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 直接读取 /dev/input/js0 的远程控制器节点
            'remote_controller_node = remote_controller.remote_controller_node:main',
            # 基于 /joy 话题的手柄转 /cmd_vel 节点，适配 ros2 run joy joy_node
            'joy_cmd_node = remote_controller.joy_cmd_node:main',
        ],
    },
)
