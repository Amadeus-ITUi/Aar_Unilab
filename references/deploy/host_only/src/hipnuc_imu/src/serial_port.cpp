#include <cstdint>
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
#include <sstream>
#include <chrono>
#include <mutex>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rcpputils/filesystem_helper.hpp>

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
		int fd = -1;
		rclcpp::TimerBase::SharedPtr publish_timer_;
		std::shared_ptr<sensor_msgs::msg::Imu> write_buffer_, read_buffer_;

		IMUNode(const std::string & node_name) : Node(node_name)
		{
			write_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
			read_buffer_ = std::make_shared<sensor_msgs::msg::Imu>();
			write_buffer_->header.frame_id = "imu_link";

			// 声明参数
			this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
			this->declare_parameter<int>("baud_rate", 115200);
			this->declare_parameter<std::string>("frame_id", "imu_link");
			this->declare_parameter<std::string>("imu_topic", "/IMU_data");
			this->declare_parameter<int>("publish_rate", 200);
			// 低通滤波参数
			this->declare_parameter<float>("imu_ang_vel_lpf_alpha", 0.4f);
			this->declare_parameter<float>("imu_gravity_lpf_alpha", 0.4f);
			// 串口超过该时间无新姿态帧则关闭并重开（运维自愈）
			this->declare_parameter<double>("serial_stale_reconnect_seconds", 0.5);
			// 数据记录和绘图参数
			this->declare_parameter<bool>("enable_plotting", false);
			this->declare_parameter<std::string>("output_file", "");

			// 获取参数
			int baud_rate, publish_rate;
			this->get_parameter("serial_port", serial_port_);
			this->get_parameter("baud_rate", baud_rate);
			this->get_parameter("frame_id", frame_id_);
			this->get_parameter("imu_topic", imu_topic_);
			this->get_parameter("publish_rate", publish_rate);
			this->get_parameter("imu_ang_vel_lpf_alpha", imu_ang_vel_lpf_alpha_);
			this->get_parameter("imu_gravity_lpf_alpha", imu_gravity_lpf_alpha_);
			this->get_parameter("serial_stale_reconnect_seconds", serial_stale_reconnect_seconds_);
			this->get_parameter("enable_plotting", enable_plotting_);
			this->get_parameter("output_file", output_file_);
			baud_rate_ = baud_rate;
			if (!std::isfinite(serial_stale_reconnect_seconds_) || serial_stale_reconnect_seconds_ <= 0.0) {
				serial_stale_reconnect_seconds_ = 0.5;
			}

			RCLCPP_INFO(this->get_logger(), "serial_port: %s", serial_port_.c_str());
			RCLCPP_INFO(this->get_logger(), "baud_rate: %d", baud_rate_);
			RCLCPP_INFO(this->get_logger(), "frame_id: %s", frame_id_.c_str());
			RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic_.c_str());
			RCLCPP_INFO(this->get_logger(), "publish_rate: %d Hz (only publish NEW serial frames; no stale republish)", publish_rate);
			RCLCPP_INFO(this->get_logger(), "serial_stale_reconnect_seconds: %.3f", serial_stale_reconnect_seconds_);
			RCLCPP_INFO(this->get_logger(), "=== IMU 滤波参数 ===");
			RCLCPP_INFO(this->get_logger(), "角速度低通滤波系数 (imu_ang_vel_lpf_alpha): %.3f", imu_ang_vel_lpf_alpha_);
			RCLCPP_INFO(this->get_logger(), "重力投影低通滤波系数 (imu_gravity_lpf_alpha): %.3f", imu_gravity_lpf_alpha_);
			RCLCPP_INFO(this->get_logger(), "滤波方法：指数移动平均 (EMA) - 新值权重=alpha, 历史权重=(1-alpha)");
			RCLCPP_INFO(this->get_logger(), "=======================");
			RCLCPP_INFO(this->get_logger(), "enable_plotting: %s", enable_plotting_ ? "true" : "false");

			write_buffer_->header.frame_id = frame_id_;

			imu_pub = this->create_publisher<sensor_msgs::msg::Imu>(imu_topic_, 10);

			auto timer_period = std::chrono::milliseconds(std::max(1, 1000 / std::max(1, publish_rate)));
			publish_timer_ = this->create_wall_timer(
				timer_period,
				std::bind(&IMUNode::publish_data, this));

			for (int i = 0; i < 3; i++) {
			    filtered_ang_vel_[i] = 0.0f;
			    filtered_projected_gravity_[i] = 0.0f;
			}
			filtered_projected_gravity_[2] = -1.0f;
			first_data_ = true;
			last_frame_time_ = std::chrono::steady_clock::now();

			if (enable_plotting_) {
			    is_recording_.store(true);
			    start_time_ = std::chrono::steady_clock::now();
			    RCLCPP_INFO(this->get_logger(), "Data recording STARTED (plotting enabled)");
			}

			if (!open_serial()) {
				RCLCPP_ERROR(this->get_logger(), "Initial serial open failed; decode thread will keep retrying");
			}
			running_.store(true);
			decode_thread_ = std::thread(&IMUNode::decode_thread, this);
		}

        ~IMUNode()
        {
            running_.store(false);
            if (decode_thread_.joinable()) {
                decode_thread_.join();
            }
            close_serial();
            
            if (enable_plotting_ && !records_.empty()) {
                RCLCPP_INFO(this->get_logger(), "Recording STOPPED (%zu frames)", records_.size());
                save_and_plot_data();
            }
        }

	private:
		bool open_serial()
		{
			std::lock_guard<std::mutex> lock(fd_mutex_);
			if (fd >= 0) {
				::close(fd);
				fd = -1;
			}

			fd = ::open(serial_port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
			if (fd < 0) {
				RCLCPP_ERROR_THROTTLE(
					this->get_logger(), *this->get_clock(), 2000,
					"Failed to open serial port: %s (errno=%d)", serial_port_.c_str(), errno);
				return false;
			}

			struct termios options;
			tcgetattr(fd, &options);
			cfmakeraw(&options);

			speed_t speed = B115200;
			for (int i = 0; baud_map[i].rate != 0; i++) {
				if (baud_map[i].rate == baud_rate_) {
					speed = baud_map[i].constant;
					break;
				}
			}
			cfsetispeed(&options, speed);
			cfsetospeed(&options, speed);

			options.c_cflag |= (CLOCAL | CREAD);
			options.c_cflag &= ~PARENB;
			options.c_cflag &= ~CSTOPB;
			options.c_cflag &= ~CSIZE;
			options.c_cflag |= CS8;
			options.c_iflag &= ~(IXON | IXOFF | IXANY);
			options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
			options.c_oflag &= ~OPOST;
			options.c_cc[VMIN] = 0;
			options.c_cc[VTIME] = 0;

			tcsetattr(fd, TCSANOW, &options);
			tcflush(fd, TCIFLUSH);
			last_frame_time_ = std::chrono::steady_clock::now();
			RCLCPP_WARN(this->get_logger(), "Serial opened: %s @ %d", serial_port_.c_str(), baud_rate_);
			return true;
		}

		void close_serial()
		{
			std::lock_guard<std::mutex> lock(fd_mutex_);
			if (fd >= 0) {
				::close(fd);
				fd = -1;
			}
		}

		int current_fd()
		{
			std::lock_guard<std::mutex> lock(fd_mutex_);
			return fd;
		}

		void decode_thread(void)
		{
			pthread_setname_np(pthread_self(), "serial_rx");
        	struct sched_param sp{}; sp.sched_priority = 60;
        	pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
            uint8_t buf[BUF_SIZE] = {0};
            WitProtocolParser parser;
            
			while (rclcpp::ok() && running_.load())
            {
			    int local_fd = current_fd();
			    if (local_fd < 0) {
			        if (!open_serial()) {
			            std::this_thread::sleep_for(std::chrono::milliseconds(500));
			        }
			        continue;
			    }

			    const double frame_age = std::chrono::duration<double>(
			        std::chrono::steady_clock::now() - last_frame_time_).count();
			    if (frame_age > serial_stale_reconnect_seconds_) {
			        RCLCPP_ERROR(
			            this->get_logger(),
			            "No new IMU attitude frame for %.3fs (threshold %.3fs); reopening serial %s",
			            frame_age, serial_stale_reconnect_seconds_, serial_port_.c_str());
			        close_serial();
			        std::this_thread::sleep_for(std::chrono::milliseconds(50));
			        open_serial();
			        continue;
			    }

			    int total_read = 0;
    		    int ret;

    		    struct timeval tv;
    		    fd_set readfds;
			    int timeout_ms = 1;
    		    while (total_read < BUF_SIZE)
    		    {
    		        FD_ZERO(&readfds);
    		        FD_SET(local_fd, &readfds);

    		        tv.tv_sec = timeout_ms / 1000;
    		        tv.tv_usec = (timeout_ms % 1000) * 1000;

    		        ret = select(local_fd + 1, &readfds, NULL, NULL, &tv);

    		        if (ret < 0)
    		        {
    		            if (errno == EINTR)
    		                continue;
    		            RCLCPP_ERROR(this->get_logger(), "select() failed errno=%d; reopening serial", errno);
    		            close_serial();
    		            break;
    		        }
    		        else if (ret == 0)
    		        {
    		            break;
    		        }

    		        ret = read(local_fd, buf + total_read, BUF_SIZE - total_read);
    		        if (ret < 0)
    		        {
    		            if (errno == EAGAIN || errno == EWOULDBLOCK)
    		                continue;
    		            RCLCPP_ERROR(this->get_logger(), "read() failed errno=%d; reopening serial", errno);
    		            close_serial();
    		            break;
    		        }
    		        else if (ret == 0)
    		        {
    		            RCLCPP_ERROR(this->get_logger(), "Serial disconnected; reopening");
    		            close_serial();
    		            break;
    		        }

    		        total_read += ret;
    		    }
                
                if(total_read > 0)
                {
                    for (int i = 0; i < total_read; i++) {
                        if (parser.process_byte(buf[i])) {
                            const WitImuData& data = parser.get_data();
                            
                            double qx, qy, qz, qw;
                            
                            if (data.has_quaternion_data) {
                                qw = data.quaternion[0];
                                qx = data.quaternion[1];
                                qy = data.quaternion[2];
                                qz = data.quaternion[3];
                            } else {
                                double roll_rad = data.angle_degree[0] * DEG_TO_RAD;
                                double pitch_rad = data.angle_degree[1] * DEG_TO_RAD;
                                double yaw_rad = data.angle_degree[2] * DEG_TO_RAD;
                                euler_to_quaternion(roll_rad, pitch_rad, yaw_rad, qx, qy, qz, qw);
                            }
                            
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
                            
                            write_buffer_->header.stamp = this->now();
                            
                            process_imu_data(qx, qy, qz, qw, 
                                           data.angular_velocity[0],
                                           data.angular_velocity[1],
                                           data.angular_velocity[2]);
                            
                            write_buffer_ = std::atomic_exchange(&read_buffer_, write_buffer_);
                            last_frame_time_ = std::chrono::steady_clock::now();
                            frame_seq_.fetch_add(1, std::memory_order_release);
                        }
                    }
                    
                    memset(buf, 0, sizeof(buf));
                }
            }
        }
        
        void process_imu_data(float qx, float qy, float qz, float qw,
                             float ang_vel_x, float ang_vel_y, float ang_vel_z) {
            // 计算重力投影（原始值）
            float projected_gravity_raw[3] = {0.0f, 0.0f, -1.0f};
            if (!(qw == 0.0f && qx == 0.0f && qy == 0.0f && qz == 0.0f)) {
                Eigen::Matrix3f R = quaternion_to_rotation_matrix(qx, qy, qz, qw);
                Eigen::Vector3f gravity_world(0.0f, 0.0f, -1.0f);
                Eigen::Vector3f projected = R.transpose() * gravity_world;
                projected_gravity_raw[0] = projected.x();
                projected_gravity_raw[1] = projected.y();
                projected_gravity_raw[2] = projected.z();
            }
            
            // 低通滤波
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
            
            // 将滤波后的数据写入IMU消息
            write_buffer_->angular_velocity_covariance[0] = filtered_projected_gravity_[0];
            write_buffer_->angular_velocity_covariance[1] = filtered_projected_gravity_[1];
            write_buffer_->angular_velocity_covariance[2] = filtered_projected_gravity_[2];
            
            // 调试：每50帧打印一次当前的重力投影（机体系）
            // static int gravity_log_counter = 0;
            // gravity_log_counter++;
            // if (gravity_log_counter % 50 == 0) {
            //     RCLCPP_INFO(this->get_logger(),
            //                 "IMU projected gravity (body frame): [%.3f, %.3f, %.3f]",
            //                 -filtered_projected_gravity_[0],
            //                 -filtered_projected_gravity_[1],
            //                 filtered_projected_gravity_[2]);
            // }
            
            // 如果正在记录，保存数据
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
            // 严格只发串口新姿态帧：没有新帧就不 publish，避免把旧数据伪装成 200Hz 活流。
            const uint64_t seq = frame_seq_.load(std::memory_order_acquire);
            if (seq == 0) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 5000,
                    "No IMU attitude frame received yet. Check serial connection and device.");
                return;
            }
            if (seq == last_published_seq_) {
                return;
            }
            last_published_seq_ = seq;

            auto data = std::atomic_load(&read_buffer_);
            imu_pub->publish(*data);
        }

        void save_and_plot_data() {
            std::lock_guard<std::mutex> lock(records_mutex_);
            
            if (records_.empty()) {
                RCLCPP_WARN(this->get_logger(), "No data recorded, skipping save and plot.");
                return;
            }
            
            // 生成输出文件名（仅文件名部分）
            std::string filename;
            if (output_file_.empty()) {
                auto now = std::chrono::system_clock::now();
                auto time_t_now = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << "imu_data_" << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S") << ".csv";
                filename = ss.str();
            } else {
                filename = output_file_;
                if (filename.find(".csv") == std::string::npos) {
                    filename += ".csv";
                }
            }
            
            // 确定保存目录：优先使用 DEPLOY_CPP_ROOT，否则使用当前工作目录
            std::string base_dir;
            const char* project_root = std::getenv("DEPLOY_CPP_ROOT");
            if (project_root) {
                base_dir = std::string(project_root);
            } else {
                rcpputils::fs::path cwd = rcpputils::fs::current_path();
                base_dir = cwd.string();
            }
            rcpputils::fs::path output_dir = rcpputils::fs::path(base_dir) / "plots" / "imu";
            try {
                rcpputils::fs::create_directories(output_dir);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to create directory %s: %s",
                             output_dir.string().c_str(), e.what());
                return;
            }
            
            rcpputils::fs::path csv_path = output_dir / rcpputils::fs::path(filename).filename();
            std::string csv_file = csv_path.string();
            
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
            
            // 调用 Python 脚本绘图，PDF 保存在同一目录下
            std::string pdf_file = csv_file;
            std::size_t dot_pos = pdf_file.rfind('.');
            if (dot_pos != std::string::npos) {
                pdf_file.replace(dot_pos, std::string::npos, ".pdf");
            } else {
                pdf_file += ".pdf";
            }
            
            RCLCPP_INFO(this->get_logger(), "Generating plots: %s", pdf_file.c_str());
            
            std::string python_script = ament_index_cpp::get_package_share_directory("hipnuc_imu") + "/scripts/plot_imu_debug.py";
            std::string command = "python3 " + python_script + " " + csv_file + " " + pdf_file;
            int result = system(command.c_str());
            
            if (result == 0) {
                RCLCPP_INFO(this->get_logger(), "Plots saved successfully: %s", pdf_file.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to generate plots (exit code: %d)", result);
            }
        }

        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
        std::thread decode_thread_;
        std::atomic<bool> running_{false};
        std::mutex fd_mutex_;
        std::string serial_port_;
        std::string frame_id_;
        std::string imu_topic_;
        int baud_rate_ = 115200;
        double serial_stale_reconnect_seconds_ = 0.5;
        std::atomic<uint64_t> frame_seq_{0};
        uint64_t last_published_seq_ = 0;
        std::chrono::steady_clock::time_point last_frame_time_{};
        
        // 滤波相关
        float imu_ang_vel_lpf_alpha_;
        float imu_gravity_lpf_alpha_;
        float filtered_ang_vel_[3];
        float filtered_projected_gravity_[3];
        bool first_data_;
        
        // 数据记录相关
        bool enable_plotting_;
        std::string output_file_;
        std::atomic<bool> is_recording_{false};
        std::vector<IMURecord> records_;
        std::mutex records_mutex_;
        std::chrono::steady_clock::time_point start_time_;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<IMUNode>("imu_node");
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
