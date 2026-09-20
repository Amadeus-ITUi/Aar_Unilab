#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

class SweepFrequencyNode : public rclcpp::Node {
public:
    SweepFrequencyNode();
    ~SweepFrequencyNode();

private:
    // 数据结构：记录每个时刻的数据
    struct DataRecord {
        double timestamp;
        std::vector<float> command_position;  // 指令位置
        std::vector<float> actual_position;   // 实际位置
        std::vector<float> actual_velocity;   // 实际速度
        std::vector<float> actual_torque;     // 实际力矩
    };

    // 参数
    std::vector<int> motor_ids_;
    float start_frequency_;      // 起始频率 (Hz)
    float end_frequency_;        // 结束频率 (Hz)
    float amplitude_;            // 幅度 (rad)
    float publish_rate_;         // 下发频率 (Hz)
    float record_rate_;          // 数据记录频率 (Hz)，用于控制CSV文件大小
    double duration_;            // 持续时间 (秒)
    std::string output_file_;    // 输出文件路径

    // 运行时变量
    std::atomic<bool> is_running_{false};
    std::chrono::steady_clock::time_point start_time_;
    double elapsed_time_{0.0};
    double current_frequency_{0.0};
    double phase_{0.0};
    
    // 数据记录
    std::vector<DataRecord> data_records_;
    std::mutex data_mutex_;
    
    // 电机状态（从反馈中获取）
    struct MotorState {
        float position{0.0f};
        float velocity{0.0f};
        float torque{0.0f};
    };
    std::map<int, MotorState> motor_states_;
    std::mutex state_mutex_;
    
    // 预分配缓冲区
    std_msgs::msg::Float32MultiArray command_msg_;
    std::vector<float> current_commands_;
    std::mutex command_mutex_;  // 保护 current_commands_ 等命令数据，支持 record_timer 并发读取
    
    // ROS2 接口
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr motor_feedback_subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr command_publisher_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    rclcpp::TimerBase::SharedPtr record_timer_;

    // 回调函数
    void motor_feedback_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void publish_command();
    void record_timer_callback();
    
    // 辅助函数
    void save_data_to_file();
    void record_current_data();  // 记录当前数据
    double calculate_current_frequency(double elapsed);
    float generate_sweep_signal(double t, double freq);
    void plot_data();  // 绘制数据图表
};

