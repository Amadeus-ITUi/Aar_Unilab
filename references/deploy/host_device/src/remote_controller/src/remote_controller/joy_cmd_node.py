#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 /joy 的手柄节点：
- 左摇杆前后 -> 线速度 x（vx）
- 右摇杆左右 -> 角速度 z（dyaw）
- 左肩键(LB) 降低高度，右肩键(RB) 抬高高度

与推理节点接口对齐：
- 发布到 `/cmd_vel`，消息类型 `geometry_msgs/Twist`
- InferenceNode 从：
    - linear.x 读取 vx
    - angular.z 读取 dyaw
    - linear.z 读取 height
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

# 与 remote_controller_node 保持一致的高度下限
MIN_HEIGHT = 0.2


class JoyCmdNode(Node):
    def __init__(self):
        super().__init__('joy_cmd_node')

        # 参数：与推理节点保持一致的范围
        self.declare_parameter('max_vx', 1.0)        # 与训练侧 vx 范围 ±1.0 保持一致
        self.declare_parameter('max_dyaw', 1.0)      # 对应 InferenceNode 中的 ±0.6 rad/s
        self.declare_parameter('max_height', 0.3)    # 对应 InferenceNode 中的高度限制
        self.declare_parameter('height_step', 0.02)  # 每次按键修改的高度增量
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('cmd_topic', '/cmd_vel')

        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_dyaw = float(self.get_parameter('max_dyaw').value)
        self.max_height = float(self.get_parameter('max_height').value)
        self.height_step = float(self.get_parameter('height_step').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)

        # 高度命令：和推理节点默认值对齐 0.23
        self.height = 0.23

        # QoS 设置：与推理节点的 cmd 订阅保持一样 (best_effort, depth=1)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            sensor_qos,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_topic,
            sensor_qos,
        )

        self.get_logger().info(
            f'JoyCmdNode 启动: /joy -> {self.cmd_topic} '
            f'(vx∈[-{self.max_vx},{self.max_vx}], dyaw∈[-{self.max_dyaw},{self.max_dyaw}], height∈[-{self.max_height},{self.max_height}])'
        )

    @staticmethod
    def _apply_deadzone(val: float, dz: float) -> float:
        return 0.0 if abs(val) < dz else val

    def joy_callback(self, msg: Joy):
        # 轴和按键索引基于实测 Flydigi Dune Fox 的 xpad 映射。
        axes = msg.axes
        buttons = msg.buttons

        # 安全检查
        if len(axes) < 2:
            return

        # 左摇杆前后：axes[1]，上推为 +1，下推为 -1；前进为正。
        raw_vx = axes[1]

        # 左摇杆左右：axes[0]，左为 +1，右为 -1；左转为正。
        raw_dyaw = axes[0]

        raw_vx = self._apply_deadzone(raw_vx, self.deadzone)
        raw_dyaw = self._apply_deadzone(raw_dyaw, self.deadzone)

        vx = max(-self.max_vx, min(self.max_vx, raw_vx * self.max_vx))
        dyaw = max(-self.max_dyaw, min(self.max_dyaw, raw_dyaw * self.max_dyaw))

        # 肩键：LB(Ros 按钮索引4) / RB(索引5)，根据你的手柄映射可微调
        lb = buttons[4] if len(buttons) > 4 else 0
        rb = buttons[5] if len(buttons) > 5 else 0

        # === 高度调整：边沿触发 + 范围限制 [MIN_HEIGHT, max_height] ===
        # 初始化上一次按钮状态
        if not hasattr(self, "_last_lb_pressed"):
            self._last_lb_pressed = False
            self._last_rb_pressed = False

        lb_pressed = bool(lb)
        rb_pressed = bool(rb)

        # 检测从“未按”到“按下”的边沿
        lb_just_pressed = lb_pressed and not self._last_lb_pressed
        rb_just_pressed = rb_pressed and not self._last_rb_pressed

        if lb_just_pressed and not rb_pressed:
            self.height -= self.height_step
        elif rb_just_pressed and not lb_pressed:
            self.height += self.height_step

        # 更新上一次状态
        self._last_lb_pressed = lb_pressed
        self._last_rb_pressed = rb_pressed

        # 限制高度范围：下限 MIN_HEIGHT，上限 max_height
        if self.height > self.max_height:
            self.height = self.max_height
        if self.height < MIN_HEIGHT:
            self.height = MIN_HEIGHT

        # 构造 Twist，与 InferenceNode::subs_cmd_callback 对齐
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.angular.z = float(dyaw)
        cmd.linear.z = float(self.height)

        self.cmd_pub.publish(cmd)

        # 可选：打印少量日志，方便调试
        # self.get_logger().debug(
        #     f"joy -> cmd: vx={cmd.linear.x:.3f}, dyaw={cmd.angular.z:.3f}, height={cmd.linear.z:.3f}"
        # )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = JoyCmdNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('JoyCmdNode 退出')
    finally:
        # 销毁节点
        if node is not None:
            node.destroy_node()
        # 避免在已由 launch 调用 rcl_shutdown 后再次 shutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # 如果上层（如 launch）已经执行过 shutdown，则忽略这里的异常
            pass


if __name__ == '__main__':
    main()
