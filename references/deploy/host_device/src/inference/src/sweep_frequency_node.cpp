#include "sweep_frequency_node.hpp"
#include <pthread.h>
#include <sched.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <algorithm>

SweepFrequencyNode::SweepFrequencyNode() : Node("sweep_frequency_node") {
    // 参数声明
    this->declare_parameter<std::vector<int64_t>>("motor_ids", std::vector<int64_t>{1, 2, 3, 4, 5, 6});
    this->declare_parameter<float>("start_frequency", 0.1f);   // 起始频率 (Hz)
    this->declare_parameter<float>("end_frequency", 5.0f);     // 结束频率 (Hz)
    this->declare_parameter<float>("amplitude", 0.3f);         // 幅度 (rad)
    this->declare_parameter<float>("publish_rate", 250.0f);    // 冻结 ESD-Link 上限 (Hz)
    this->declare_parameter<float>("record_rate", 250.0f);     // 数据记录频率 (Hz)
    this->declare_parameter<double>("duration", 40.0);         // 持续时间 (秒)
    this->declare_parameter<std::string>("output_file", "");   // 输出文件路径（为空则自动生成）

    // 获取参数
    std::vector<int64_t> ids = this->get_parameter("motor_ids").as_integer_array();
    motor_ids_.assign(ids.begin(), ids.end());
    this->get_parameter("start_frequency", start_frequency_);
    this->get_parameter("end_frequency", end_frequency_);
    this->get_parameter("amplitude", amplitude_);
    this->get_parameter("publish_rate", publish_rate_);
    this->get_parameter("record_rate", record_rate_);
    this->get_parameter("duration", duration_);
    this->get_parameter("output_file", output_file_);

    // 参数验证
    if (start_frequency_ <= 0.0f || end_frequency_ <= 0.0f || start_frequency_ > end_frequency_) {
        RCLCPP_FATAL(this->get_logger(), "Invalid frequency range: [%.2f, %.2f] Hz", 
                    start_frequency_, end_frequency_);
        rclcpp::shutdown();
        return;
    }
    if (amplitude_ <= 0.0f) {
        RCLCPP_FATAL(this->get_logger(), "Invalid amplitude: %.3f rad", amplitude_);
        rclcpp::shutdown();
        return;
    }
    if (publish_rate_ <= 0.0f || publish_rate_ > 250.0f) {
        RCLCPP_FATAL(this->get_logger(), "Invalid publish rate: %.1f Hz (ESD-Link range: 0 < rate <= 250)", publish_rate_);
        rclcpp::shutdown();
        return;
    }
    if (record_rate_ <= 0.0f) {
        RCLCPP_FATAL(this->get_logger(), "Invalid record rate: %.1f Hz", record_rate_);
        rclcpp::shutdown();
        return;
    }
    if (record_rate_ > publish_rate_) {
        RCLCPP_WARN(
            this->get_logger(),
            "record_rate %.1f Hz > publish_rate %.1f Hz, clamping record_rate to publish_rate",
            record_rate_, publish_rate_);
        record_rate_ = publish_rate_;
    }
    if (duration_ <= 0.0) {
        RCLCPP_FATAL(this->get_logger(), "Invalid duration: %.2f s", duration_);
        rclcpp::shutdown();
        return;
    }

    // 设置默认保存目录（使用当前工作目录或环境变量）
    std::string plots_dir;
    const char* project_root = std::getenv("DEPLOY_CPP_ROOT");
    if (project_root) {
        plots_dir = std::string(project_root) + "/plots";
    } else {
        // 使用当前工作目录的 plots 子目录
        plots_dir = (std::filesystem::current_path() / "plots").string();
    }
    
    // 如果没有指定输出文件，自动生成到plots目录
    if (output_file_.empty()) {
        // 确保plots目录存在
        try {
            std::filesystem::create_directories(plots_dir);
        } catch (const std::filesystem::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "无法创建目录 %s: %s", plots_dir.c_str(), e.what());
            RCLCPP_ERROR(this->get_logger(), "请检查目录权限或设置环境变量 DEPLOY_CPP_ROOT 指定项目根目录");
            rclcpp::shutdown();
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        
        ss << "sweep_frequency_";
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
        ss << ".csv";
        output_file_ = (std::filesystem::path(plots_dir) / ss.str()).string();
    } else {
        // 如果指定了相对路径，将其转换为绝对路径并放在plots目录下
        std::filesystem::path output_path(output_file_);
        if (output_path.is_relative()) {
            try {
                std::filesystem::create_directories(plots_dir);
                output_file_ = (std::filesystem::path(plots_dir) / output_path.filename()).string();
            } catch (const std::filesystem::filesystem_error& e) {
                RCLCPP_ERROR(this->get_logger(), "无法创建目录 %s: %s", plots_dir.c_str(), e.what());
                rclcpp::shutdown();
                return;
            }
        } else {
            // 如果指定了绝对路径，确保其父目录存在
            try {
                std::filesystem::create_directories(output_path.parent_path());
            } catch (const std::filesystem::filesystem_error& e) {
                RCLCPP_ERROR(this->get_logger(), "无法创建目录 %s: %s", output_path.parent_path().c_str(), e.what());
                rclcpp::shutdown();
                return;
            }
        }
    }

    // 初始化命令缓冲区
    current_commands_.resize(motor_ids_.size(), 0.0f);
    command_msg_.data.resize(motor_ids_.size(), 0.0f);
    
    // 初始化电机状态
    for (int id : motor_ids_) {
        motor_states_[id] = {0.0f, 0.0f, 0.0f};
    }

    // QoS配置（与推理节点一致）
    auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

    // 订阅电机反馈
    motor_feedback_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/policy/joint_states", sensor_data_qos,
        std::bind(&SweepFrequencyNode::motor_feedback_callback, this, std::placeholders::_1));

    // 发布扫频命令
    command_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "/policy/commands", control_command_qos);

    // 创建定时器（使用微秒精度，避免频率误差）
    auto timer_period = std::chrono::microseconds(static_cast<long long>(1000000.0 / publish_rate_));
    publish_timer_ = this->create_wall_timer(
        timer_period,
        std::bind(&SweepFrequencyNode::publish_command, this));

    // 创建记录定时器：record_rate <= publish_rate。
    auto record_period = std::chrono::microseconds(static_cast<long long>(1000000.0 / record_rate_));
    record_timer_ = this->create_wall_timer(
        record_period,
        std::bind(&SweepFrequencyNode::record_timer_callback, this));

    // 记录开始时间
    start_time_ = std::chrono::steady_clock::now();
    is_running_.store(true);

    // 输出参数信息
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "扫频曲线下发节点已启动");
    RCLCPP_INFO(this->get_logger(), "电机ID: [%s]", 
               [&](){
                   std::string ids_str;
                   for(size_t i = 0; i < motor_ids_.size(); i++){
                       if(i > 0) ids_str += ", ";
                       ids_str += std::to_string(motor_ids_[i]);
                   }
                   return ids_str;
               }().c_str());
    RCLCPP_INFO(this->get_logger(), "频率范围: %.2f Hz -> %.2f Hz", start_frequency_, end_frequency_);
    RCLCPP_INFO(this->get_logger(), "幅度: %.3f rad", amplitude_);
    RCLCPP_INFO(this->get_logger(), "下发频率: %.1f Hz", publish_rate_);
    RCLCPP_INFO(this->get_logger(), "记录频率: %.1f Hz", record_rate_);
    RCLCPP_INFO(this->get_logger(), "持续时间: %.2f 秒", duration_);
    RCLCPP_INFO(this->get_logger(), "输出文件: %s", output_file_.c_str());
    RCLCPP_INFO(this->get_logger(), "========================================");
}

SweepFrequencyNode::~SweepFrequencyNode() {
    // 停止发布
    is_running_.store(false);
    
    // 保存数据
    RCLCPP_INFO(this->get_logger(), "正在保存数据到文件: %s", output_file_.c_str());
    save_data_to_file();
    RCLCPP_INFO(this->get_logger(), "数据已保存，共 %zu 条记录", data_records_.size());
}

void SweepFrequencyNode::motor_feedback_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (!msg || !is_running_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // 解析电机反馈（假设关节名称格式为 "motor_{id}"）
    for (size_t i = 0; i < msg->name.size(); ++i) {
        std::string name = msg->name[i];
        // 提取电机ID（从 "motor_1" 中提取 "1"）
        if (name.find("motor_") == 0) {
            try {
                int motor_id = std::stoi(name.substr(6));  // "motor_".length() = 6
                if (motor_states_.count(motor_id)) {
                    motor_states_[motor_id].position = i < msg->position.size() ? msg->position[i] : 0.0f;
                    motor_states_[motor_id].velocity = i < msg->velocity.size() ? msg->velocity[i] : 0.0f;
                    motor_states_[motor_id].torque = i < msg->effort.size() ? msg->effort[i] : 0.0f;
                }
            } catch (const std::exception& e) {
                // 忽略解析错误
            }
        }
    }
}

void SweepFrequencyNode::publish_command() {
    if (!is_running_.load()) {
        return;
    }

    // 计算经过的时间
    auto now = std::chrono::steady_clock::now();
    elapsed_time_ = std::chrono::duration<double>(now - start_time_).count();

    // 检查是否超时
    if (elapsed_time_ >= duration_) {
        RCLCPP_INFO(this->get_logger(), "预设时间已到 (%.2f 秒)，准备关闭节点并保存数据...", duration_);
        is_running_.store(false);
        
        // 最后一次记录数据
        record_current_data();
        
        // 停止定时器并关闭节点
        publish_timer_->cancel();
        rclcpp::shutdown();
        return;
    }

    // 计算当前频率（线性扫频）
    current_frequency_ = calculate_current_frequency(elapsed_time_);
    
    // 生成扫频信号
    float signal_value = generate_sweep_signal(elapsed_time_, current_frequency_);

    // 为所有电机生成命令：左侧电机(1,2,3)为负，右侧电机(4,5,6)为正。
    // 注意：record_timer_callback() 会并发读取 current_commands_，因此需要加锁
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            if (i < 3) {
                current_commands_[i] = -signal_value;
                command_msg_.data[i] = -signal_value;
            } else {
                current_commands_[i] = signal_value;
                command_msg_.data[i] = signal_value;
            }
        }
        // 保持原有“强制某些电机命令为 0”的测试逻辑
        // current_commands_[0] = 0.0f;
        // command_msg_.data[0] = 0.0f;
        current_commands_[1] = 0.0f;
        command_msg_.data[1] = 0.0f;
        current_commands_[2] = 0.0f;
        command_msg_.data[2] = 0.0f;
        current_commands_[3] = 0.0f;
        command_msg_.data[3] = 0.0f;
        current_commands_[4] = 0.0f;
        command_msg_.data[4] = 0.0f;
        current_commands_[5] = 0.0f;
        command_msg_.data[5] = 0.0f;
    }

    // 发布命令
    if (command_publisher_) {
        command_publisher_->publish(command_msg_);
    }

    // 定期输出进度信息
    static int progress_counter = 0;
    progress_counter++;
    if (progress_counter % static_cast<int>(publish_rate_) == 0) {  // 每秒输出一次
        double progress = (elapsed_time_ / duration_) * 100.0;
        RCLCPP_INFO(this->get_logger(), 
                   "进度: %.1f%% | 时间: %.2f/%.2f s | 当前频率: %.3f Hz | 信号值: %.4f",
                   progress, elapsed_time_, duration_, current_frequency_, signal_value);
    }
}

void SweepFrequencyNode::record_timer_callback() {
    if (!is_running_.load()) {
        return;
    }
    record_current_data();
}

void SweepFrequencyNode::record_current_data() {
    DataRecord record;
    // 用当前时刻计算 timestamp，支持 record_rate > publish_rate（允许记录重复值但 timestamp 仍连续增长）
    auto now = std::chrono::steady_clock::now();
    record.timestamp = std::chrono::duration<double>(now - start_time_).count();
    
    // 保存指令位置
    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        record.command_position = current_commands_;
    }
    
    // 保存实际状态（需要加锁读取）
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        record.actual_position.resize(motor_ids_.size());
        record.actual_velocity.resize(motor_ids_.size());
        record.actual_torque.resize(motor_ids_.size());
        
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            int id = motor_ids_[i];
            if (motor_states_.count(id)) {
                record.actual_position[i] = motor_states_[id].position;
                record.actual_velocity[i] = motor_states_[id].velocity;
                record.actual_torque[i] = motor_states_[id].torque;
            } else {
                record.actual_position[i] = 0.0f;
                record.actual_velocity[i] = 0.0f;
                record.actual_torque[i] = 0.0f;
            }
        }
    }
    
    // 保存记录
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        data_records_.push_back(record);
    }
}

double SweepFrequencyNode::calculate_current_frequency(double elapsed) {
    // 线性扫频：f(t) = f_start + (f_end - f_start) * (t / duration)
    double ratio = std::min(elapsed / duration_, 1.0);
    return start_frequency_ + (end_frequency_ - start_frequency_) * ratio;
}

float SweepFrequencyNode::generate_sweep_signal(double t, double freq) {
    // 线性扫频的正弦信号
    // phase = ∫f(t)dt = f_start * t + 0.5 * (f_end - f_start) * t^2 / duration
    // 对于线性扫频，瞬时相位为：
    static double last_t = 0.0;
    static double last_phase = 0.0;
    
    if (t < last_t) {
        // 重置
        last_t = 0.0;
        last_phase = 0.0;
    }
    
    // 计算相位增量（使用梯形积分）
    double dt = t - last_t;
    double phase_increment = last_t == 0.0 ? 
        start_frequency_ * dt :  // 第一次，使用起始频率
        (current_frequency_ + (calculate_current_frequency(last_t))) * 0.5 * dt;  // 梯形积分
    
    phase_ = last_phase + phase_increment;
    last_phase = phase_;
    last_t = t;
    
    // 生成正弦信号
    return amplitude_ * std::sin(2.0 * M_PI * phase_);
}

void SweepFrequencyNode::save_data_to_file() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (data_records_.empty()) {
        RCLCPP_WARN(this->get_logger(), "没有数据需要保存");
        return;
    }

    std::ofstream file(output_file_);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "无法打开文件进行写入: %s", output_file_.c_str());
        return;
    }

    // 写入CSV头部
    file << "timestamp";
    for (size_t i = 0; i < motor_ids_.size(); ++i) {
        file << ",motor_" << motor_ids_[i] << "_cmd_pos";
        file << ",motor_" << motor_ids_[i] << "_act_pos";
        file << ",motor_" << motor_ids_[i] << "_act_vel";
        file << ",motor_" << motor_ids_[i] << "_act_torque";
    }
    file << "\n";

    // 设置精度
    file << std::fixed << std::setprecision(6);

    // 写入数据
    for (const auto& record : data_records_) {
        file << record.timestamp;
        for (size_t i = 0; i < motor_ids_.size(); ++i) {
            file << "," << (i < record.command_position.size() ? record.command_position[i] : 0.0f);
            file << "," << (i < record.actual_position.size() ? record.actual_position[i] : 0.0f);
            file << "," << (i < record.actual_velocity.size() ? record.actual_velocity[i] : 0.0f);
            file << "," << (i < record.actual_torque.size() ? record.actual_torque[i] : 0.0f);
        }
        file << "\n";
    }

    file.close();
    RCLCPP_INFO(this->get_logger(), "数据已成功保存到: %s (共 %zu 条记录)", 
               output_file_.c_str(), data_records_.size());
    
    // 自动绘制数据图表
    plot_data();
}

void SweepFrequencyNode::plot_data() {
    // 尝试使用Python的ament_index查找脚本路径
    std::string find_cmd = "python3 -c \"from ament_index_python import get_package_share_directory; import os; print(os.path.join(get_package_share_directory('inference'), 'scripts', 'plot_sweep_frequency.py'))\" 2>/dev/null";
    
    FILE* pipe = popen(find_cmd.c_str(), "r");
    std::string script_path;
    
    if (pipe) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            script_path = buffer;
            // 移除换行符和空白字符
            script_path.erase(std::remove_if(script_path.begin(), script_path.end(), 
                [](char c) { return c == '\n' || c == '\r' || c == ' ' || c == '\t'; }), 
                script_path.end());
        }
        pclose(pipe);
    }
    
    // 如果通过ament_index找不到，尝试其他路径
    if (script_path.empty() || !std::filesystem::exists(script_path)) {
        // 尝试从环境变量构造路径
        const char* ament_prefix = std::getenv("AMENT_PREFIX_PATH");
        if (ament_prefix) {
            std::string ament_path = ament_prefix;
            std::istringstream iss(ament_path);
            std::string path;
            while (std::getline(iss, path, ':')) {
                std::string test_path = path + "/inference/share/inference/scripts/plot_sweep_frequency.py";
                if (std::filesystem::exists(test_path)) {
                    script_path = test_path;
                    break;
                }
            }
        }
    }
    
    // 如果还是找不到，尝试相对路径（用于开发环境）
    if (script_path.empty() || !std::filesystem::exists(script_path)) {
        std::string test_path = std::string(std::filesystem::current_path().string()) + "/src/inference/scripts/plot_sweep_frequency.py";
        if (std::filesystem::exists(test_path)) {
            script_path = test_path;
        }
    }
    
    if (script_path.empty() || !std::filesystem::exists(script_path)) {
        RCLCPP_WARN(this->get_logger(), "无法找到绘图脚本 plot_sweep_frequency.py，跳过绘图");
        RCLCPP_WARN(this->get_logger(), "请确保脚本已安装到: <install_dir>/share/inference/scripts/plot_sweep_frequency.py");
        return;
    }
    
    // 查找Python解释器（优先使用isaac_gym conda环境中的Python）
    std::string python_cmd = "python3";  // 默认使用python3
    const char* conda_env = std::getenv("CONDA_DEFAULT_ENV");
    const char* conda_prefix = std::getenv("CONDA_PREFIX");
    
    // 如果当前在isaac_gym环境中，使用CONDA_PREFIX中的Python
    if (conda_env && std::string(conda_env) == "isaac_gym" && conda_prefix) {
        std::string conda_python = std::string(conda_prefix) + "/bin/python";
        if (std::filesystem::exists(conda_python)) {
            python_cmd = conda_python;
            RCLCPP_INFO(this->get_logger(), "使用当前激活的 isaac_gym conda 环境中的 Python: %s", python_cmd.c_str());
        }
    }
    // 如果CONDA_PREFIX存在但不是isaac_gym，尝试查找isaac_gym环境
    else if (conda_prefix) {
        // 从CONDA_PREFIX推断conda根目录
        std::filesystem::path conda_path(conda_prefix);
        std::filesystem::path conda_root = conda_path.parent_path();  // 通常是 /home/user/miniconda3
        
        // 尝试查找isaac_gym环境
        std::string isaac_gym_python = (conda_root / "envs" / "isaac_gym" / "bin" / "python").string();
        if (std::filesystem::exists(isaac_gym_python)) {
            python_cmd = isaac_gym_python;
            RCLCPP_INFO(this->get_logger(), "找到 isaac_gym conda 环境中的 Python: %s", python_cmd.c_str());
        }
    }
    // 如果CONDA_PREFIX不存在，尝试常见的conda环境路径
    else {
        const char* home = std::getenv("HOME");
        if (home) {
            std::vector<std::string> possible_paths = {
                std::string(home) + "/anaconda3/envs/isaac_gym/bin/python",
                std::string(home) + "/miniconda3/envs/isaac_gym/bin/python",
                std::string(home) + "/.conda/envs/isaac_gym/bin/python",
                "/opt/conda/envs/isaac_gym/bin/python",
            };
            
            for (const auto& path : possible_paths) {
                if (std::filesystem::exists(path)) {
                    python_cmd = path;
                    RCLCPP_INFO(this->get_logger(), "找到 isaac_gym conda 环境中的 Python: %s", python_cmd.c_str());
                    break;
                }
            }
        }
    }
    
    // 如果设置了PYTHON环境变量，使用它（优先级最高）
    const char* python_env = std::getenv("PYTHON");
    if (python_env && std::filesystem::exists(python_env)) {
        python_cmd = python_env;
        RCLCPP_INFO(this->get_logger(), "使用环境变量PYTHON指定的解释器: %s", python_cmd.c_str());
    }
    
    // 构建输出PDF文件名（与CSV文件在同一目录）
    std::filesystem::path csv_path(output_file_);
    std::string output_pdf = (csv_path.parent_path() / (csv_path.stem().string() + "_plot.pdf")).string();
    
    // 构建命令（使用绝对路径并转义特殊字符）
    std::string cmd = "'" + python_cmd + "' '" + script_path + "' '" + output_file_ + "' -o '" + output_pdf + "'";
    
    RCLCPP_INFO(this->get_logger(), "正在绘制数据图表...");
    RCLCPP_DEBUG(this->get_logger(), "执行命令: %s", cmd.c_str());
    
    // 执行Python脚本
    int ret = system(cmd.c_str());
    if (ret == 0) {
        if (std::filesystem::exists(output_pdf)) {
            RCLCPP_INFO(this->get_logger(), "图表已成功保存到: %s", output_pdf.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "绘图脚本执行成功，但未找到输出文件: %s", output_pdf.c_str());
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "绘图脚本执行失败，返回码: %d", ret);
        RCLCPP_WARN(this->get_logger(), "使用的Python解释器: %s", python_cmd.c_str());
        RCLCPP_WARN(this->get_logger(), "请确保该Python环境中已安装 matplotlib 和 numpy");
        RCLCPP_WARN(this->get_logger(), "如果使用 isaac_gym 环境，请激活环境后安装: conda activate isaac_gym && pip install matplotlib numpy");
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SweepFrequencyNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
