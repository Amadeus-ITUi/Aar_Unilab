#include <iostream>
#include <sensor_msgs/msg/imu.hpp>
#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <thread>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifdef __cplusplus
extern "C"{
#endif
#include <poll.h>
#ifdef __cplusplus
}
#endif

#define GRA_ACC     (9.8)
#define DEG_TO_RAD  (M_PI / 180.0)
#define BUF_SIZE (256)

// WIT IMU 协议常量
#define WIT_HEADER      0x55
#define WIT_ACC_TYPE    0x51
#define WIT_GYR_TYPE    0x52
#define WIT_ANG_TYPE    0x53
#define WIT_MAG_TYPE    0x54
#define WIT_QUA_TYPE    0x59  // 四元数数据包
#define WIT_PACKET_SIZE 11

using namespace std::chrono_literals;
using namespace std;

static const struct {
    int rate;
    speed_t constant;
} baud_map[] = {
    {4800, B4800}, {9600, B9600}, {19200, B19200}, {38400, B38400},
    {57600, B57600}, {115200, B115200}, {230400, B230400}, {460800, B460800}, {921600, B921600},
    {0, B0}  // Sentinel
};

// WIT IMU 数据结构
struct WitImuData {
    float acceleration[3] = {0.0f, 0.0f, 0.0f};
    float angular_velocity[3] = {0.0f, 0.0f, 0.0f};
    float angle_degree[3] = {0.0f, 0.0f, 0.0f};
    float magnetometer[3] = {0.0f, 0.0f, 0.0f};
    float quaternion[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // w, x, y, z
    bool has_angle_data = false;
    bool has_quaternion_data = false;
};

// 数据记录结构
struct IMURecord {
    double timestamp;  // 秒
    float ang_vel[3];  // 角速度 x, y, z (滤波后)
    float ang_vel_raw[3];  // 角速度 x, y, z (原始)
    float projected_gravity[3];  // 重力投影 x, y, z (滤波后)
    float projected_gravity_raw[3];  // 重力投影 x, y, z (原始)
    float quaternion[4];  // 四元数 w, x, y, z
};

// WIT 协议解析器
class WitProtocolParser {
private:
    uint8_t buff[WIT_PACKET_SIZE];
    int key = 0;
    WitImuData imu_data_;

    // 将两个字节转换为有符号短整型（小端序）
    int16_t hex_to_short(uint8_t low, uint8_t high) {
        return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
    }

    // 校验和检查
    bool check_sum(uint8_t* data, int len, uint8_t checksum) {
        uint8_t sum = 0;
        for (int i = 0; i < len; i++) {
            sum += data[i];
        }
        return (sum & 0xFF) == checksum;
    }

public:
    // 处理一个字节，返回 true 表示解析到完整数据包
    bool process_byte(uint8_t byte) {
        buff[key] = byte;
        key++;

        // 检查起始字节
        if (buff[0] != WIT_HEADER) {
            key = 0;
            return false;
        }

        // 等待完整数据包
        if (key < WIT_PACKET_SIZE) {
            return false;
        }

        // 数据包完整，开始解析
        bool angle_flag = false;
        uint8_t data_type = buff[1];

        // 校验和检查
        if (!check_sum(buff, 10, buff[10])) {
            key = 0;
            return false;
        }

        // 根据数据类型解析
        switch (data_type) {
            case WIT_ACC_TYPE:  // 加速度计
                imu_data_.acceleration[0] = hex_to_short(buff[2], buff[3]) / 32768.0f * 16.0f * GRA_ACC;
                imu_data_.acceleration[1] = hex_to_short(buff[4], buff[5]) / 32768.0f * 16.0f * GRA_ACC;
                imu_data_.acceleration[2] = hex_to_short(buff[6], buff[7]) / 32768.0f * 16.0f * GRA_ACC;
                break;

            case WIT_GYR_TYPE:  // 陀螺仪
                imu_data_.angular_velocity[0] = hex_to_short(buff[2], buff[3]) / 32768.0f * 2000.0f * DEG_TO_RAD;
                imu_data_.angular_velocity[1] = hex_to_short(buff[4], buff[5]) / 32768.0f * 2000.0f * DEG_TO_RAD;
                imu_data_.angular_velocity[2] = hex_to_short(buff[6], buff[7]) / 32768.0f * 2000.0f * DEG_TO_RAD;
                break;

            case WIT_ANG_TYPE:  // 角度
                imu_data_.angle_degree[0] = hex_to_short(buff[2], buff[3]) / 32768.0f * 180.0f;
                imu_data_.angle_degree[1] = hex_to_short(buff[4], buff[5]) / 32768.0f * 180.0f;
                imu_data_.angle_degree[2] = hex_to_short(buff[6], buff[7]) / 32768.0f * 180.0f;
                angle_flag = true;
                imu_data_.has_angle_data = true;
                break;

            case WIT_MAG_TYPE:  // 磁力计
                imu_data_.magnetometer[0] = hex_to_short(buff[2], buff[3]) / 10.0f;
                imu_data_.magnetometer[1] = hex_to_short(buff[4], buff[5]) / 10.0f;
                imu_data_.magnetometer[2] = hex_to_short(buff[6], buff[7]) / 10.0f;
                break;

            case WIT_QUA_TYPE:  // 四元数
                imu_data_.quaternion[0] = hex_to_short(buff[2], buff[3]) / 32768.0f;  // qw
                imu_data_.quaternion[1] = hex_to_short(buff[4], buff[5]) / 32768.0f;  // qx
                imu_data_.quaternion[2] = hex_to_short(buff[6], buff[7]) / 32768.0f;  // qy
                imu_data_.quaternion[3] = hex_to_short(buff[8], buff[9]) / 32768.0f;  // qz
                angle_flag = true;  // 四元数也可作为姿态数据触发发布
                imu_data_.has_quaternion_data = true;
                break;

            default:
                key = 0;
                return false;
        }

        key = 0;
        return angle_flag;  // 只有角度数据包返回 true，触发发布
    }

    const WitImuData& get_data() const {
        return imu_data_;
    }
};

// 从欧拉角计算四元数
void euler_to_quaternion(double roll, double pitch, double yaw, 
                         double& qx, double& qy, double& qz, double& qw) {
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);

    qw = cr * cp * cy + sr * sp * sy;
    qx = sr * cp * cy - cr * sp * sy;
    qy = cr * sp * cy + sr * cp * sy;
    qz = cr * cp * sy - sr * sp * cy;
}

// 从四元数计算旋转矩阵
Eigen::Matrix3f quaternion_to_rotation_matrix(float qx, float qy, float qz, float qw) {
    // 归一化
    float norm = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if(norm == 0.0f) {
        return Eigen::Matrix3f::Identity();
    }
    qx /= norm; qy /= norm; qz /= norm; qw /= norm;
    
    float xx = qx*qx, yy = qy*qy, zz = qz*qz;
    float xy = qx*qy, xz = qx*qz, yz = qy*qz;
    float wx = qw*qx, wy = qw*qy, wz = qw*qz;
    
    Eigen::Matrix3f R;
    R << 1.0f - 2.0f*(yy + zz), 2.0f*(xy - wz), 2.0f*(xz + wy),
         2.0f*(xy + wz), 1.0f - 2.0f*(xx + zz), 2.0f*(yz - wx),
         2.0f*(xz - wy), 2.0f*(yz + wx), 1.0f - 2.0f*(xx + yy);
    return R;
}

class IMUNode : public rclcpp::Node
{
	public:
		int fd = 0;
		rclcpp::TimerBase::SharedPtr publish_timer_;
		IMUNode() : Node("imu_node")	
		{
			this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
			this->declare_parameter<int>("baud_rate", 115200);
			this->declare_parameter<std::string>("frame_id", "imu_link");
			this->declare_parameter<std::string>("imu_topic", "/IMU_data");
            this->declare_parameter<int>("publish_rate", 100);
            this->declare_parameter<bool>("debug_mode", false);
            this->declare_parameter<bool>("enable_plotting", false);
            this->declare_parameter<float>("imu_ang_vel_lpf_alpha", 1.0f);
            this->declare_parameter<float>("imu_gravity_lpf_alpha", 1.0f);
            this->declare_parameter<std::string>("output_file", "");

			this->get_parameter("serial_port", serial_port);
			this->get_parameter("baud_rate", baud_rate);
			this->get_parameter("frame_id", frame_id);
			this->get_parameter("imu_topic", imu_topic);
            this->get_parameter("publish_rate", publish_rate);
            this->get_parameter("debug_mode", debug_mode_);
            this->get_parameter("enable_plotting", enable_plotting_);
            this->get_parameter("imu_ang_vel_lpf_alpha", imu_ang_vel_lpf_alpha_);
            this->get_parameter("imu_gravity_lpf_alpha", imu_gravity_lpf_alpha_);
            this->get_parameter("output_file", output_file_);

			RCLCPP_INFO(this->get_logger(),"serial_port: %s", serial_port.c_str());
			RCLCPP_INFO(this->get_logger(), "baud_rate: %d", baud_rate);
			RCLCPP_INFO(this->get_logger(), "frame_id: %s", frame_id.c_str());
			RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "publish_rate: %d Hz", publish_rate);
            RCLCPP_INFO(this->get_logger(), "debug_mode: %s", debug_mode_ ? "true" : "false");
            
            if (debug_mode_) {
                RCLCPP_INFO(this->get_logger(), "Debug mode enabled:");
                RCLCPP_INFO(this->get_logger(), "  - enable_plotting: %s", enable_plotting_ ? "true" : "false");
                RCLCPP_INFO(this->get_logger(), "  - imu_ang_vel_lpf_alpha: %.3f", imu_ang_vel_lpf_alpha_);
                RCLCPP_INFO(this->get_logger(), "  - imu_gravity_lpf_alpha: %.3f", imu_gravity_lpf_alpha_);
                RCLCPP_INFO(this->get_logger(), "滤波方法：指数移动平均 (EMA) - 新值权重=alpha, 历史权重=(1-alpha)");
                
                // 如果启用绘图，自动开始记录
                if (enable_plotting_) {
                    is_recording_.store(true);
                    start_time_ = std::chrono::steady_clock::now();
                    RCLCPP_INFO(this->get_logger(), "Recording STARTED (plotting enabled)");
                }
            }
            
			write_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
			write_buffer_->header.frame_id = frame_id;
			read_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
			read_buffer_->header.frame_id = frame_id;
			auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
			imu_pub = this->create_publisher<sensor_msgs::msg::Imu>(imu_topic, sensor_data_qos);

			fd = open_serial(serial_port, baud_rate);

            if (fd > 0) {
                decode_thread_ = std::thread(&IMUNode::decode_thread, this);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to open serial port");
            }

			publish_timer_ = this->create_wall_timer(
			    std::chrono::milliseconds(1000 / publish_rate),
			    std::bind(&IMUNode::publish_data, this)
			);
			
			// 初始化滤波状态
			for (int i = 0; i < 3; i++) {
			    filtered_ang_vel_[i] = 0.0f;
                
			    filtered_projected_gravity_[i] = 0.0f;
			}
			filtered_projected_gravity_[2] = -1.0f;  // 默认重力向下
			first_data_ = true;
			start_time_ = std::chrono::steady_clock::now();
		}

        ~IMUNode()
        {
            running_.store(false);
            
            if (decode_thread_.joinable()) {
                decode_thread_.join();
            }
            
            if (fd > 0) {
                close(fd);
            }
            
            // 保存数据并绘图 (仅在启用绘图时)
            if (debug_mode_ && enable_plotting_ && !records_.empty()) {
                RCLCPP_INFO(this->get_logger(), "Recording STOPPED (%zu frames)", records_.size());
                save_and_plot_data();
            }
        }

	private:
		void decode_thread(void)
		{
			pthread_setname_np(pthread_self(), "serial_rx");
        	struct sched_param sp{}; sp.sched_priority = 60;
        	pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
            uint8_t buf[BUF_SIZE] = {0};
            WitProtocolParser parser;
            
			while(running_.load() && rclcpp::ok())
            {
			    int total_read = 0;
    		    int ret;

    		    // Setup for select() timeout
    		    struct timeval tv;
    		    fd_set readfds;
			    int timeout_ms = 1;
    		    while (total_read < BUF_SIZE)
    		    {
    		        // Reset select() parameters for each iteration
    		        FD_ZERO(&readfds);
    		        FD_SET(fd, &readfds);

    		        // Configure timeout for this iteration
    		        tv.tv_sec = timeout_ms / 1000;
    		        tv.tv_usec = (timeout_ms % 1000) * 1000;

    		        // Wait for data or timeout
    		        ret = select(fd + 1, &readfds, NULL, NULL, &tv);

    		        if (ret < 0)
    		        {
    		            // Handle interruption by signal
    		            if (errno == EINTR)
    		                continue;
    		            perror("select");
    		            return;
    		        }
    		        else if (ret == 0)
    		        {
    		            // No data received within timeout period
    		            break;
    		        }

    		        // Data is available, read it
    		        ret = read(fd, buf + total_read, BUF_SIZE - total_read);
    		        if (ret < 0)
    		        {
    		            // Handle non-blocking operations
    		            if (errno == EAGAIN || errno == EWOULDBLOCK)
    		                continue;
    		            perror("read");
    		            return;
    		        }
    		        else if (ret == 0)
    		        {
    		            // Port closed or disconnected
    		            break;
    		        }

    		        // Update total bytes read
    		        total_read += ret;
    		    }
                
                if(total_read > 0)
                {
                    static int angle_packets = 0;
                    // 逐个字节处理 WIT 协议
                    for (int i = 0; i < total_read; i++) {
                        if (parser.process_byte(buf[i])) {
                            // 收到姿态数据包，更新 IMU 消息
                            angle_packets++;
                            if (angle_packets <= 3) {
                                RCLCPP_INFO(this->get_logger(), "Decoded orientation packet #%d", angle_packets);
                            }
                            const WitImuData& data = parser.get_data();
                            
                            double qx, qy, qz, qw;
                            
                            // 优先使用四元数数据，如果没有则从欧拉角转换
                            if (data.has_quaternion_data) {
                                // 直接使用 IMU 输出的四元数
                                qw = data.quaternion[0];
                                qx = data.quaternion[1];
                                qy = data.quaternion[2];
                                qz = data.quaternion[3];
                                
                                if (angle_packets <= 3) {
                                    RCLCPP_INFO(this->get_logger(), "Using QUATERNION data: qw=%.3f, qx=%.3f, qy=%.3f, qz=%.3f", 
                                               qw, qx, qy, qz);
                                }
                            } else {
                                // 从欧拉角转换四元数（兼容旧固件）
                                double roll_rad = data.angle_degree[0] * DEG_TO_RAD;
                                double pitch_rad = data.angle_degree[1] * DEG_TO_RAD;
                                double yaw_rad = data.angle_degree[2] * DEG_TO_RAD;
                                euler_to_quaternion(roll_rad, pitch_rad, yaw_rad, qx, qy, qz, qw);
                                
                                if (angle_packets <= 3) {
                                    RCLCPP_INFO(this->get_logger(), "Using EULER angles: roll=%.1f°, pitch=%.1f°, yaw=%.1f°", 
                                               data.angle_degree[0], data.angle_degree[1], data.angle_degree[2]);
                                }
                            }
                            
                            // 更新消息
                            write_buffer_->orientation.w = qw;
                            write_buffer_->orientation.x = qx;
                            write_buffer_->orientation.y = qy;
                            write_buffer_->orientation.z = qz;
                            
                            write_buffer_->angular_velocity.x = data.angular_velocity[0];
                            write_buffer_->angular_velocity.y = data.angular_velocity[1];
                            write_buffer_->angular_velocity.z = data.angular_velocity[2];
                            
                            write_buffer_->linear_acceleration.x = data.acceleration[0];
                            write_buffer_->linear_acceleration.y = data.acceleration[1];
                            write_buffer_->linear_acceleration.z = data.acceleration[2];
                            
                            write_buffer_->header.stamp = rclcpp::Clock().now();
                            
                            // Debug 模式：计算重力投影和滤波
                            if (debug_mode_) {
                                process_imu_for_debug(qx, qy, qz, qw, 
                                                     data.angular_velocity[0],
                                                     data.angular_velocity[1],
                                                     data.angular_velocity[2]);
                            }
                            
                            write_buffer_ = std::atomic_exchange(&read_buffer_, write_buffer_);
                        }
                    }
                    
                    memset(buf, 0, sizeof(buf));
                }
            }
        }
        
        void process_imu_for_debug(float qx, float qy, float qz, float qw,
                                   float ang_vel_x, float ang_vel_y, float ang_vel_z) {
            // 计算重力投影（原始值，与推理节点一致）
            float projected_gravity_raw[3] = {0.0f, 0.0f, -1.0f};
            if (!(qw == 0.0f && qx == 0.0f && qy == 0.0f && qz == 0.0f)) {
                Eigen::Matrix3f R = quaternion_to_rotation_matrix(qx, qy, qz, qw);
                Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
                Eigen::Vector3f projected = R.transpose() * gravity_world;
                projected_gravity_raw[0] = projected.x();
                projected_gravity_raw[1] = projected.y();
                projected_gravity_raw[2] = projected.z();
            }
            
            // 角速度低通滤波（与推理节点一致）
            if (first_data_) {
                filtered_ang_vel_[0] = ang_vel_x;
                filtered_ang_vel_[1] = ang_vel_y;
                filtered_ang_vel_[2] = ang_vel_z;
                filtered_projected_gravity_[0] = projected_gravity_raw[0];
                filtered_projected_gravity_[1] = projected_gravity_raw[1];
                filtered_projected_gravity_[2] = projected_gravity_raw[2];
                first_data_ = false;
            } else {
                float alpha_vel = imu_ang_vel_lpf_alpha_;
                filtered_ang_vel_[0] = alpha_vel * ang_vel_x + (1.0f - alpha_vel) * filtered_ang_vel_[0];
                filtered_ang_vel_[1] = alpha_vel * ang_vel_y + (1.0f - alpha_vel) * filtered_ang_vel_[1];
                filtered_ang_vel_[2] = alpha_vel * ang_vel_z + (1.0f - alpha_vel) * filtered_ang_vel_[2];
                
                float alpha_grav = imu_gravity_lpf_alpha_;
                filtered_projected_gravity_[0] = alpha_grav * projected_gravity_raw[0] + (1.0f - alpha_grav) * filtered_projected_gravity_[0];
                filtered_projected_gravity_[1] = alpha_grav * projected_gravity_raw[1] + (1.0f - alpha_grav) * filtered_projected_gravity_[1];
                filtered_projected_gravity_[2] = alpha_grav * projected_gravity_raw[2] + (1.0f - alpha_grav) * filtered_projected_gravity_[2];
            }
            
            // 如果正在记录，保存数据（包括原始值和滤波值）
            if (is_recording_.load()) {
                auto now = std::chrono::steady_clock::now();
                double timestamp = std::chrono::duration<double>(now - start_time_).count();
                
                IMURecord record;
                record.timestamp = timestamp;
                // 滤波后的值
                record.ang_vel[0] = filtered_ang_vel_[0];
                record.ang_vel[1] = filtered_ang_vel_[1];
                record.ang_vel[2] = filtered_ang_vel_[2];
                record.projected_gravity[0] = filtered_projected_gravity_[0];
                record.projected_gravity[1] = filtered_projected_gravity_[1];
                record.projected_gravity[2] = filtered_projected_gravity_[2];
                // 原始值
                record.ang_vel_raw[0] = ang_vel_x;
                record.ang_vel_raw[1] = ang_vel_y;
                record.ang_vel_raw[2] = ang_vel_z;
                record.projected_gravity_raw[0] = projected_gravity_raw[0];
                record.projected_gravity_raw[1] = projected_gravity_raw[1];
                record.projected_gravity_raw[2] = projected_gravity_raw[2];
                // 四元数
                record.quaternion[0] = qw;
                record.quaternion[1] = qx;
                record.quaternion[2] = qy;
                record.quaternion[3] = qz;
                
                std::lock_guard<std::mutex> lock(records_mutex_);
                records_.push_back(record);
            }
        }

        void publish_data(void)
        {
            auto data = std::atomic_load(&read_buffer_);
            
            // 检查数据是否有效（时间戳不为0表示有数据更新过）
            static bool first_publish = true;
            if (first_publish && data->header.stamp.sec == 0 && data->header.stamp.nanosec == 0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "No IMU data received yet. Check serial connection and device.");
                first_publish = false;
            }
            
            imu_pub->publish(*data);
        }
        
        void save_and_plot_data() {
            std::lock_guard<std::mutex> lock(records_mutex_);
            
            if (records_.empty()) {
                RCLCPP_WARN(this->get_logger(), "No data recorded, skipping save and plot.");
                return;
            }
            
            // 生成输出文件名
            std::string csv_file;
            if (output_file_.empty()) {
                auto now = std::chrono::system_clock::now();
                auto time_t_now = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << "imu_debug_" << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S") << ".csv";
                csv_file = ss.str();
            } else {
                csv_file = output_file_;
                if (csv_file.find(".csv") == std::string::npos) {
                    csv_file += ".csv";
                }
            }
            
            // 保存 CSV
            std::ofstream file(csv_file);
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", csv_file.c_str());
                return;
            }
            
            file << "timestamp,ang_vel_x,ang_vel_y,ang_vel_z,ang_vel_x_raw,ang_vel_y_raw,ang_vel_z_raw,"
                 << "gravity_x,gravity_y,gravity_z,gravity_x_raw,gravity_y_raw,gravity_z_raw,"
                 << "quat_w,quat_x,quat_y,quat_z\n";
            for (const auto& record : records_) {
                file << std::fixed << std::setprecision(6) << record.timestamp << ","
                     << record.ang_vel[0] << "," << record.ang_vel[1] << "," << record.ang_vel[2] << ","
                     << record.ang_vel_raw[0] << "," << record.ang_vel_raw[1] << "," << record.ang_vel_raw[2] << ","
                     << record.projected_gravity[0] << "," << record.projected_gravity[1] << "," 
                     << record.projected_gravity[2] << ","
                     << record.projected_gravity_raw[0] << "," << record.projected_gravity_raw[1] << "," 
                     << record.projected_gravity_raw[2] << ","
                     << record.quaternion[0] << "," << record.quaternion[1] << "," 
                     << record.quaternion[2] << "," << record.quaternion[3] << "\n";
            }
            file.close();
            
            RCLCPP_INFO(this->get_logger(), "Saved %zu records to %s", records_.size(), csv_file.c_str());
            
            // 调用 Python 脚本绘图
            std::string pdf_file = csv_file;
            size_t pos = pdf_file.find(".csv");
            if (pos != std::string::npos) {
                pdf_file.replace(pos, 4, ".pdf");
            }
            
            std::string script_path =
                ament_index_cpp::get_package_share_directory("hipnuc_imu") + "/scripts/plot_imu_debug.py";
            std::string command = "python3 " + script_path + " " + csv_file + " " + pdf_file;
            
            RCLCPP_INFO(this->get_logger(), "Generating plots: %s", pdf_file.c_str());
            int result = system(command.c_str());
            
            if (result == 0) {
                RCLCPP_INFO(this->get_logger(), "Plots saved successfully: %s", pdf_file.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to generate plots (exit code: %d)", result);
            }
        }

        int open_serial(std::string port, int baud) {
            const char* port_device = port.c_str();
            
            // 检查设备文件是否存在
            if (access(port_device, F_OK) != 0) {
                RCLCPP_ERROR(this->get_logger(), "Serial port device not found: %s", port_device);
                RCLCPP_INFO(this->get_logger(), "Please check:");
                RCLCPP_INFO(this->get_logger(), "  1. USB device is connected");
                RCLCPP_INFO(this->get_logger(), "  2. Run: ls -la /dev/ttyUSB* to find available devices");
                RCLCPP_INFO(this->get_logger(), "  3. If device exists but not accessible, check permissions");
                RCLCPP_INFO(this->get_logger(), "  4. If CH340 device detected but no /dev/ttyUSB*, disable brltty:");
                RCLCPP_INFO(this->get_logger(), "     sudo systemctl stop brltty-udev.service");
                RCLCPP_INFO(this->get_logger(), "     sudo systemctl disable brltty-udev.service");
                return -1;
            }
            
            int fd = open(port_device, O_RDWR | O_NOCTTY | O_NDELAY);

            if (fd == -1) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open serial port: %s", port_device);
                perror("unable to open serial port");
                if (errno == EACCES) {
                    RCLCPP_ERROR(this->get_logger(), "Permission denied. Try: sudo chmod 666 %s", port_device);
                    RCLCPP_INFO(this->get_logger(), "Or add user to dialout group: sudo usermod -a -G dialout $USER");
                }
                return -1;
            }

            struct termios options;
            memset(&options, 0, sizeof(options));
            tcgetattr(fd, &options);

            // Set baud rate
            speed_t baud_constant = B0;
            for (int i = 0; baud_map[i].rate != 0; i++) {
                if (baud_map[i].rate == baud) {
                    baud_constant = baud_map[i].constant;
    		        break;
                }
            }

            if (baud_constant == B0) {
                fprintf(stderr, "Unsupported baud rate: %d\n", baud);
    		    return -1;
            }

            if (cfsetispeed(&options, baud_constant) < 0 || 
    		    cfsetospeed(&options, baud_constant) < 0) {
    		    perror("Error setting baud rate");
    		    return -1;
    		}

			 // Configure other port settings
    		options.c_cflag &= ~PARENB;  // No parity
    		options.c_cflag &= ~CSTOPB;  // 1 stop bit
    		options.c_cflag &= ~CSIZE;
    		options.c_cflag |= CS8;      // 8 data bits
    		options.c_cflag |= (CLOCAL | CREAD);  // Enable receiver, ignore modem control lines
		
    		// Disable hardware flow control
    		options.c_cflag &= ~CRTSCTS;
		
    		// Set input mode (non-canonical, no echo,...)
    		options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    		options.c_iflag &= ~(IXON | IXOFF | IXANY);  // Disable software flow control
    		options.c_iflag &= ~(INLCR | ICRNL);  // Disable newline & carriage return translation
		
    		// Set output mode (raw output)
    		options.c_oflag &= ~OPOST;
		
    		// Set read timeout and minimum character count
    		options.c_cc[VMIN] = 0;  // Minimum number of characters
    		options.c_cc[VTIME] = 0;  // Timeout in deciseconds
		
    		// Apply the new settings
    		if (tcsetattr(fd, TCSANOW, &options) != 0) {
    		    perror("Error setting port attributes");
    		    return -1;
    		}
		
    		// Flush the buffer
    		tcflush(fd, TCIOFLUSH);

			return fd;
        }

        std::string serial_port;
        int baud_rate;
        int publish_rate;
        std::string frame_id;
        std::string imu_topic;
		std::shared_ptr<sensor_msgs::msg::Imu> write_buffer_, read_buffer_;
        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
        std::thread decode_thread_;
        std::atomic<bool> running_{true};
        
        // Debug 模式相关
        bool debug_mode_;
        bool enable_plotting_;  // 是否启用数据记录和绘图
        float imu_ang_vel_lpf_alpha_;
        float imu_gravity_lpf_alpha_;
        std::string output_file_;
        
        // 滤波状态
        float filtered_ang_vel_[3];
        float filtered_projected_gravity_[3];
        bool first_data_;
        
        // 数据记录（debug模式下自动开始记录）
        std::atomic<bool> is_recording_{false};
        std::vector<IMURecord> records_;
        std::mutex records_mutex_;
        std::chrono::steady_clock::time_point start_time_;
};


int main(int argc, const char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<IMUNode>());
	rclcpp::shutdown();

	return 0;
}
