#!/usr/bin/env python3
"""
简单键盘控制节点（与 inference 模块命令一致）：
- 发布 `geometry_msgs/msg/Twist` 到 `/cmd_vel`
- 对应 inference 中的命令含义：
  - linear.x : vx   （前向线速度）
  - angular.z: dyaw （绕 z 轴角速度）
  - linear.z : height（高度命令）

按键说明：
  w/s : x 方向前进 / 后退（vx）
  q/e : 左转 / 右转（dyaw）
  空格: 增加 height
  z   : 减少 height   （终端无法检测裸 Shift，使用 z 代替）
  x   : 停止（vx=0, dyaw=0）
  h   : 打印帮助

退出：Ctrl+C
cd ~/project/deploy_cpp/inference
source /opt/ros/humble/setup.bash
source install/setup.sh
python3 src/inference/scripts/keyboard_cmd_vel.py
"""

import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP_TEXT = """
键盘控制 /cmd_vel （按 q 退出）:

  w/s : x 方向 前进 / 后退 (vx)
  q/e : 左转 / 右转 （dyaw）
  空格: 增加 height
  z   : 减少 height
  x   : 停止 (vx=0, dyaw=0)
  h   : 打印本帮助

当前命令：vx=%.2f  dyaw=%.2f  height=%.2f
"""


def get_key(settings):
    """非阻塞读取单个按键，超时时间 0.1 秒。

    如果当前 stdin 不是 TTY（例如被 launch 重定向），直接返回空字符串。
    """
    if settings is None or not sys.stdin.isatty():
        return ''

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardCmdVel(Node):
    def __init__(self):
        super().__init__('keyboard_cmd_vel')

        # 最大线速度、角速度和高度命令范围（可通过参数覆盖）
        self.declare_parameter('max_lin_vel', 0.3)    # m/s，对应 inference 中 clamp [-0.3,0.3]
        self.declare_parameter('max_ang_vel', 0.6)    # rad/s，对应 clamp [-0.6,0.6]
        self.declare_parameter('min_height', 0.2)     # 最小高度命令
        self.declare_parameter('max_height', 0.3)     # 最大高度命令
        self.declare_parameter('height_step', 0.02)   # 每次按键改变的高度步长

        self.max_lin_vel = float(self.get_parameter('max_lin_vel').value)
        self.max_ang_vel = float(self.get_parameter('max_ang_vel').value)
        self.min_height = float(self.get_parameter('min_height').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.height_step = float(self.get_parameter('height_step').value)

        self.pub_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # 20Hz 定时器：既用来轮询按键，也保持持续发布当前命令
        self.timer_ = self.create_timer(0.05, self.timer_cb)

        self.current_lin = 0.0      # vx
        self.current_ang = 0.0      # dyaw
        # 初始高度设置为默认值 0.25
        self.current_height = 0.25  # height (linear.z)

        # 保存终端设置，后面恢复
        if sys.stdin.isatty():
            self.settings_ = termios.tcgetattr(sys.stdin)
        else:
            # 在非 TTY 环境（例如被 launch 重定向 stdin）时，不做键盘读取，
            # 但节点仍然可以正常运行（持续发布当前命令）。
            self.settings_ = None
            self.get_logger().warn(
                "当前 stdin 不是终端（TTY），无法读取键盘输入；"
                "如果需要键盘控制，请在普通终端中运行该节点。"
            )

        self.get_logger().info(
            "键盘控制节点已启动，焦点保持在该终端窗口。\n"
            "按 h 显示帮助，Ctrl+C 退出。"
        )

    def timer_cb(self):
        # 读取按键（非阻塞）
        key = get_key(self.settings_)

        if key == 'w':
            self.current_lin = self.max_lin_vel
        elif key == 's':
            self.current_lin = -self.max_lin_vel
        elif key == 'q':
            self.current_ang = self.max_ang_vel
        elif key == 'e':
            self.current_ang = -self.max_ang_vel
        elif key == ' ':
            # 增加高度
            self.current_height += self.height_step
            if self.current_height > self.max_height:
                self.current_height = self.max_height
        elif key == 'z':
            # 减少高度
            self.current_height -= self.height_step
            if self.current_height < self.min_height:
                self.current_height = self.min_height
        elif key == 'x':
            self.current_lin = 0.0
            self.current_ang = 0.0
            self.current_height = 0.25
        elif key == 'h':
            print(HELP_TEXT % (self.current_lin, self.current_ang, self.current_height))
        elif key != '':
            # 其他键忽略
            pass

        # 发布 Twist
        twist = Twist()
        twist.linear.x = self.current_lin     # vx
        twist.angular.z = self.current_ang    # dyaw
        twist.linear.z = self.current_height  # height
        self.pub_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardCmdVel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前恢复终端设置，并发送一次停止命令
        try:
            if getattr(node, "settings_", None) is not None and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings_)
        except Exception:
            pass
        stop = Twist()
        try:
            node.pub_.publish(stop)
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()

