#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远程控制器节点
使用Xbox手柄控制机器人，发布高层指令（线速度和角速度）

本节点设计为在「纯终端 / 无图形界面」环境下运行，因此：
- 使用 SDL 的 dummy 视频驱动，避免依赖真实显示设备
- 仅使用 joystick 输入，不创建窗口
"""

import os
import pygame
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist

# 为了在无图形界面的终端环境下正常使用 pygame 的事件系统，
# 强制使用 SDL 的 dummy 视频驱动，避免 "video system not initialized" 错误。
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# 速度范围：与 inference_node.cpp 对齐
MAX_VX = 1.0      # 前后移动速度范围 ±1.0
MAX_DYAW = 0.6    # 旋转角速度范围 ±0.6
MAX_HEIGHT = 0.3  # 高度上限
MIN_HEIGHT = 0.2  # 高度下限
HEIGHT_STEP = 0.02  # 每次按键修改的高度增量

# 跳跃高度命令范围（与新策略中 cmd 第四维 jump_height 对齐，通常 0~1）
MAX_JUMP_HEIGHT = 1.0
MIN_JUMP_HEIGHT = 0.0


class Buttons:
    """替代原来的Buttons类，用于存储按钮状态"""
    def __init__(self):
        self.A = False
        self.B = False
        self.X = False
        self.Y = False
        self.LB = False
        self.RB = False
        self.up = False
        self.down = False
        
    def update(self, A, B, X, Y, LB, RB, up, down):
        self.A = A
        self.B = B
        self.X = X
        self.Y = Y
        self.LB = LB
        self.RB = RB
        self.up = up
        self.down = down
        
    def __str__(self):
        return f"Buttons(A={self.A}, B={self.B}, X={self.X}, Y={self.Y}, LB={self.LB}, RB={self.RB}, up={self.up}, down={self.down})"
    
    def __repr__(self):
        return self.__str__()


class XBoxController:
    """Xbox手柄控制器类（简化版，不包含线程）

    说明：
    - 不再依赖蓝牙配对逻辑
    - 直接使用 Linux joystick 设备 `/dev/input/js0`
      （在大多数系统中，索引 0 对应 js0）
    """
    def __init__(self, logger=None):
        self.logger = logger

        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0

        # 初始化 pygame（包括 joystick 和一个最小的 dummy display）
        pygame.init()
        # 在 dummy 视频驱动下创建一个最小窗口，保证 event 子系统可用
        try:
            if not pygame.display.get_init():
                pygame.display.init()
            if pygame.display.get_surface() is None:
                pygame.display.set_mode((1, 1))
        except Exception as e:
            # 在极端环境下，display 初始化失败也不致命，只是不能用 event 系统
            if logger:
                logger.warning(f"pygame display 初始化失败: {e}")
        
        # 初始化并等待手柄连接（直接使用 js0 / 索引 0）
        self.p1 = self._wait_for_joystick()

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LB_pressed = False
        self.RB_pressed = False

        self.buttons = Buttons()
        
        # 高度命令：与 inference_node.cpp reset() 中的默认值对齐 0.23
        self.height = 0.23
        # 跳跃高度命令：新策略 cmd 第四维，对应 jump_height
        self.jump_height = 0.0

    def _wait_for_joystick(self):
        """
        直接使用 /dev/input/js0 对应的手柄设备（索引 0）

        不再做蓝牙配对/名称匹配，只要系统中存在一个 joystick 设备，
        就使用第 0 个设备。一般情况下这就是 /dev/input/js0。
        """
        pygame.joystick.init()
        log_func = self.logger.info if self.logger else print

        while True:
            pygame.joystick.quit()
            pygame.joystick.init()
            num_joysticks = pygame.joystick.get_count()

            if num_joysticks == 0:
                # 这里不再提示“蓝牙手柄”，而是直接提示 js0 设备
                if self.logger:
                    self.logger.warn("未检测到任何手柄设备（/dev/input/js0）。请检查 USB 接收器或有线手柄是否已插好。")
                else:
                    print("未检测到任何手柄设备（/dev/input/js0），请插入手柄...", end="\r")
                time.sleep(0.5)
                continue

            # 只要检测到至少 1 个 joystick，就直接使用索引 0
            try:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                name = joystick.get_name()
                axes = joystick.get_numaxes()
                buttons = joystick.get_numbuttons()
                log_func(f"使用手柄设备索引 0（通常对应 /dev/input/js0）: {name} - {axes} 轴, {buttons} 按钮")
                return joystick
            except Exception as e:
                if self.logger:
                    self.logger.error(f"初始化手柄设备索引 0 失败: {e}")
                else:
                    print(f"初始化手柄设备索引 0 失败: {e}")
                time.sleep(0.5)

    def get_velocity_command(self):
        """
        获取速度命令（线速度、角速度、高度和跳跃高度）
        返回: (vx, dyaw, height, jump_height)
        与 inference_node.cpp 接口对齐：
        - linear.x 读取 vx (范围 ±1.0)
        - angular.z 读取 dyaw (范围 ±0.6)
        - linear.z 读取 height (范围 0.2-0.3)
        """
        # 必须先调用 pump() 来更新手柄状态
        pygame.event.pump()
        
        # 读取摇杆值（在pump之后读取，确保状态是最新的）
        # Flydigi/xpad 实测：axis 1 上推为 +1，作为正向前进。
        l_y = self.p1.get_axis(1)  # 左摇杆 Y（前后）
        # Flydigi/xpad 实测：左摇杆 axis 0 左为 +1，作为正向左转。
        r_x = self.p1.get_axis(0)  # 左摇杆 X（左右旋转）

        # 摇杆死区设置（去除小抖动）
        deadzone = 0.1
        if abs(l_y) < deadzone:
            l_y = 0.0
        if abs(r_x) < deadzone:
            r_x = 0.0

        # 应用速度范围限制
        vx = max(-MAX_VX, min(MAX_VX, l_y * MAX_VX))
        dyaw = max(-MAX_DYAW, min(MAX_DYAW, r_x * MAX_DYAW))

        # 直接查询按钮的当前状态
        # 注意：pygame 按钮索引可能与 ROS2 joy 不同
        # 根据 joy_cmd_node.py，ROS2 joy 使用 buttons[4] 和 buttons[5] 对应 LB 和 RB
        # 但 pygame 的按钮索引可能不同，需要根据实际手柄调整
        self.A_pressed = self.p1.get_button(0)  # A button
        self.B_pressed = self.p1.get_button(1)  # B button
        self.X_pressed = self.p1.get_button(2)  # X button
        self.Y_pressed = self.p1.get_button(3)  # Y button
        
        # 获取按钮总数
        num_buttons = self.p1.get_numbuttons()
        
        # 尝试多个可能的按钮索引（pygame 的按钮索引可能因手柄而异）
        # 常见映射：LB=4, RB=5 (对应 ROS2 joy 的 buttons[4] 和 buttons[5])
        # 但也可能是：LB=6, RB=7
        self.LB_pressed = False
        self.RB_pressed = False
        
        # 先尝试索引 4 和 5（对应 ROS2 joy 的 buttons[4] 和 buttons[5]）
        if num_buttons > 4:
            self.LB_pressed = self.p1.get_button(4)
        if num_buttons > 5:
            self.RB_pressed = self.p1.get_button(5)
        
        # 如果 4/5 没有反应，尝试 6/7（某些手柄映射）
        if not self.LB_pressed and num_buttons > 6:
            self.LB_pressed = self.p1.get_button(6)
        if not self.RB_pressed and num_buttons > 7:
            self.RB_pressed = self.p1.get_button(7)

        # 按下肩键时调整高度（边沿触发：只在按钮从释放到按下的瞬间触发一次）
        # 使用实例变量记录上次按钮状态
        if not hasattr(self, '_last_lb_pressed'):
            self._last_lb_pressed = False
            self._last_rb_pressed = False
        
        # 检测按钮从释放到按下的边沿触发
        lb_just_pressed = self.LB_pressed and not self._last_lb_pressed
        rb_just_pressed = self.RB_pressed and not self._last_rb_pressed
        
        if lb_just_pressed and not self.RB_pressed:
            self.height -= HEIGHT_STEP
            if self.logger:
                self.logger.info(f"LB按下: 高度减少到 {self.height:.3f}")
        elif rb_just_pressed and not self.LB_pressed:
            self.height += HEIGHT_STEP
            if self.logger:
                self.logger.info(f"RB按下: 高度增加到 {self.height:.3f}")
        
        # 更新上次按钮状态
        self._last_lb_pressed = self.LB_pressed
        self._last_rb_pressed = self.RB_pressed

        # 限制高度范围（与 inference_node.cpp 的 clamp 对齐）
        if self.height > MAX_HEIGHT:
            self.height = MAX_HEIGHT
        if self.height < MIN_HEIGHT:
            self.height = MIN_HEIGHT
        # 使用 A 按钮作为简单的跳跃触发：按下时 jump_height=1.0，否则为 0.0
        # 如需更复杂的映射（如模拟量），可改为使用扳机轴。
        self.jump_height = 1.0 if self.A_pressed else 0.0
        # 限制跳跃高度范围
        if self.jump_height > MAX_JUMP_HEIGHT:
            self.jump_height = MAX_JUMP_HEIGHT
        if self.jump_height < MIN_JUMP_HEIGHT:
            self.jump_height = MIN_JUMP_HEIGHT

        return (vx, dyaw, self.height, self.jump_height)


class RemoteControllerNode(Node):
    """远程控制器ROS2节点"""
    
    def __init__(self):
        super().__init__('remote_controller_node')
        
        # 声明参数
        self.declare_parameter('command_rate', 20.0)  # 命令发布频率 Hz
        self.declare_parameter('command_topic', '/cmd_vel')  # 命令话题（改为与inference节点兼容）
        
        # 获取参数
        self.command_rate = self.get_parameter('command_rate').value
        self.command_topic = self.get_parameter('command_topic').value
        
        self.get_logger().info(f'初始化远程控制器节点')
        self.get_logger().info(f'命令发布频率: {self.command_rate} Hz')
        self.get_logger().info(f'命令话题: {self.command_topic}')
        
        # 初始化手柄控制器
        try:
            self.controller = XBoxController(logger=self.get_logger())
            self.get_logger().info('手柄控制器初始化成功')
        except Exception as e:
            self.get_logger().error(f'手柄控制器初始化失败: {e}')
            raise
        
        # QoS配置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 创建发布者
        self.cmd_pub = self.create_publisher(
            Twist,
            self.command_topic,
            qos_profile
        )
        self.get_logger().info(f'发布者已创建: {self.command_topic} -> Twist')
        
        # 创建定时器（替代线程）
        self.command_timer = self.create_timer(
            1.0 / self.command_rate,
            self.command_timer_callback
        )
        self.get_logger().info('定时器已创建')
        
        # 统计信息
        self.publish_count = 0
        self.last_stats_time = time.time()
        self.last_height = 0.23  # 用于检测高度变化
        
        # 打印按钮映射信息（帮助调试）
        if hasattr(self.controller, 'p1'):
            num_buttons = self.controller.p1.get_numbuttons()
            self.get_logger().info(f'手柄按钮总数: {num_buttons}')
            self.get_logger().info('按钮映射: A=0, B=1, X=2, Y=3, LB=4/6, RB=5/7')
        
        self.get_logger().info('远程控制器节点启动成功')
    
    def command_timer_callback(self):
        """定时器回调函数：读取手柄并发布命令"""
        print("command_timer_callback")
        try:
            # 获取速度命令 (vx, dyaw, height, jump_height)
            vx, dyaw, height, jump_height = self.controller.get_velocity_command()
            
            # 创建Twist消息，与 InferenceNode::subs_cmd_callback 对齐
            twist_msg = Twist()
            twist_msg.linear.x = float(vx)         # 前后移动速度（上推为正）
            # 使用 linear.y 传递 jump_height（与 InferenceNode::subs_cmd_callback 对齐）
            twist_msg.linear.y = float(jump_height)
            twist_msg.linear.z = float(height)     # 高度命令
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = float(dyaw)      # 左转为正
            
            # 发布消息
            self.cmd_pub.publish(twist_msg)
            self.publish_count += 1
            
            # 检测高度变化并打印
            if abs(height - self.last_height) > 0.001:
                self.get_logger().info(
                    f'高度变化: {self.last_height:.3f} -> {height:.3f} | '
                    f'LB={self.controller.LB_pressed}, RB={self.controller.RB_pressed}'
                )
                self.last_height = height
            
            # 定期打印统计信息
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:
                self.get_logger().debug(
                    f'已发布 {self.publish_count} 条命令 | '
                    f'vx={vx:.3f}, dyaw={dyaw:.3f}, height={height:.3f}, jump_h={jump_height:.3f} | '
                    f'LB={self.controller.LB_pressed}, RB={self.controller.RB_pressed}, A={self.controller.A_pressed}'
                )
                self.last_stats_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'发布命令时出错: {e}')


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = RemoteControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info('接收到终止信号')
    except Exception as e:
        print(f'节点运行出错: {e}')
    finally:
        # 确保节点被销毁，但避免重复调用 rcl_shutdown
        if node is not None:
            node.destroy_node()
        # 只有在还未 shutdown 时才调用，防止 "rcl_shutdown already called" 报错
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # 如果上层（如 launch）已经调用过 shutdown，则忽略这里的异常
            pass


if __name__ == "__main__":
    main()
