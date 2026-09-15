/**
 * @file
 * This file implements functions to receive
 * and transmit CAN frames via SocketCAN.
 */

#include "SocketCAN.hpp"
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <chrono>
#include <thread>

std::shared_ptr<spdlog::logger> SocketCAN::logger_ = nullptr;
std::unordered_map<std::string, std::shared_ptr<SocketCAN>> SocketCAN::instances_;

SocketCAN::SocketCAN(std::string interface)
    : interface_(interface), sockfd_(INIT_FD), receiving_(false) {
    open(interface);
}

SocketCAN::~SocketCAN() { this->close(); }

void SocketCAN::open(std::string interface) {
    sockfd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sockfd_ == INIT_FD) {
        logger_->error("Failed to create CAN socket");
        return;
    }

    int bufsize = 1024 * 1024;  // 1MB
    setsockopt(sockfd_, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    strncpy(if_request_.ifr_name, interface.c_str(), IFNAMSIZ);
    if (ioctl(sockfd_, SIOCGIFINDEX, &if_request_) == -1) {
        logger_->error("Unable to detect CAN interface {}", interface);

        this->close();
        return;
    }

    // Bind the socket to the network interface
    addr_.can_family = AF_CAN;
    addr_.can_ifindex = if_request_.ifr_ifindex;
    int rc = ::bind(sockfd_, reinterpret_cast<struct sockaddr *>(&addr_), sizeof(addr_));
    if (rc == -1) {
        logger_->error("Failed to bind socket to network interface {}", interface);
        this->close();
        return;
    }

    int flags = fcntl(sockfd_, F_GETFL, 0);
    if (flags == -1) {
        logger_->error("Failed to get socket flags");
        this->close();
        return;
    }
    if (fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        logger_->error("Failed to set socket to non-blocking");
        this->close();
        return;
    }

    receiving_ = true;
    receiver_thread_ = std::thread([this]() {
        pthread_setname_np(pthread_self(), "can_rx");
        struct sched_param sp{}; sp.sched_priority = 80;
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

        fd_set descriptors;
        int maxfd = sockfd_;
        struct timeval timeout;
        can_frame rx_frame;

        while (receiving_) {
            FD_ZERO(&descriptors);
            FD_SET(sockfd_, &descriptors);

            timeout.tv_sec = TIMEOUT_SEC;
            timeout.tv_usec = TIMEOUT_USEC;

            if (::select(maxfd + 1, &descriptors, NULL, NULL, &timeout) == 1) {
                while (true){
                    int len = ::read(sockfd_, &rx_frame, CAN_MTU);
                    if (len < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break; 
                        }
                        static int read_error_count = 0;
                        if (++read_error_count % 1000 == 0) {
                            logger_->warn("CAN read error: {}", strerror(errno));
                        }
                        break;
                    }
                    if (len == 0){
                        break;
                    }
                    
                    // [新增] 嗅探逻辑：如果开启了嗅探，把数据拷贝一份到嗅探缓冲区
                    if (is_spying_.load()) {
                        std::lock_guard<std::mutex> spy_lock(spy_mutex_);
                        spy_buffer_.push_back(rx_frame);
                    }
                    
                    CanCbkFunc callback_to_run;
                    {
                        std::lock_guard<std::mutex> lock(can_callback_mutex_);
                        bool is_extended = (rx_frame.can_id & CAN_EFF_FLAG) != 0;
                        uint32_t can_id = is_extended ? (rx_frame.can_id & CAN_EFF_MASK) : (rx_frame.can_id & CAN_SFF_MASK);
                        
                        uint16_t motor_id;
                        if (is_extended) {
                            // RobStride 反馈帧：[Type][Status][MotorID][MasterID]
                            // MotorID 位于 bit 8-15，和内核过滤器保持一致。
                            motor_id = (can_id >> 8) & 0xFF;
                        } else {
                            motor_id = can_id & 0x7FF; // Standard frame: 11-bit ID
                        }
                        
                        auto it = can_callback_list_.find(motor_id);
                        if (it != can_callback_list_.end()) {
                            // 检查帧类型是否匹配
                            if (it->second.is_extended == is_extended) {
                                callback_to_run = it->second.callback;
                            }
                        }
                    }
                    if (callback_to_run) {
                        callback_to_run(rx_frame);
                    }
                }
            }
        }
    });
}

void SocketCAN::close() {
    receiving_ = false;
    if (receiver_thread_.joinable()) receiver_thread_.join();

    if (sockfd_ != INIT_FD) ::close(sockfd_);
    sockfd_ = INIT_FD;
}

/**
 * @brief 直接写入 CAN 帧，零延迟、零拷贝
 * 
 * 对于 400Hz 控制，必须直接写入而不是通过队列和线程
 * 在 400Hz 控制下，最新的指令永远比旧指令重要
 * 如果发送缓冲区满，选择丢包而不是阻塞（下一帧 2.5ms 后就来了）
 */
bool SocketCAN::transmit(const can_frame &frame) {
    if (sockfd_ == INIT_FD) {
        logger_->error("Unable to transmit: Socket not open");
        return false;
    }

    // 直接写入 Socket，不经过队列
    ssize_t nbytes = ::write(sockfd_, &frame, sizeof(can_frame));

    if (nbytes < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 发送缓冲区满：在 RL 场景下，我们选择丢包而不是阻塞
            // 因为下一帧控制指令 2.5ms 后就来了，阻塞会导致整个控制回路延迟累积
            // 避免日志刷屏：首次提示，之后每 100000 次提示一次。
            static int drop_count = 0;
            ++drop_count;
            if (drop_count == 1 || drop_count % 100000 == 0) {
                logger_->warn("CAN TX buffer full, {} frames dropped; check CAN bus state/load", drop_count);
            }
        } else {
            // 记录严重错误，但不抛出异常中断控制流
            static int tx_error_count = 0;
            ++tx_error_count;
            if (tx_error_count == 1 || tx_error_count % 100000 == 0) {
                logger_->error("CAN write error: {}", strerror(errno));
            }
        }
        return false;
    }

    if (nbytes != static_cast<ssize_t>(sizeof(can_frame))) {
        static int partial_write_count = 0;
        if (++partial_write_count % 1000 == 0) {
            logger_->warn("CAN partial write, {} of {} bytes written ({} times)",
                          nbytes, sizeof(can_frame), partial_write_count);
        }
        return false;
    }

    return true;
}

void SocketCAN::add_can_callback(const CanCbkFunc callback, const CanCbkId id, bool is_extended) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_[id] = {callback, is_extended};
    update_can_filters_locked();
}

void SocketCAN::remove_can_callback(const CanCbkId id) {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.erase(id);
    update_can_filters_locked();
}

void SocketCAN::clear_can_callbacks() {
    std::lock_guard<std::mutex> lock(can_callback_mutex_);
    can_callback_list_.clear();
    update_can_filters_locked();
}

/**
 * @brief 安全接收所有帧（使用嗅探模式，避免与 receiver_thread_ 竞争）
 * 
 * 不再直接调用 read()，而是让 receiver_thread_ 帮我们读取
 * receiver_thread_ 会在读取时检查 is_spying_ 标志，如果开启则拷贝数据到 spy_buffer_
 * 
 * @param timeout_ms 等待时间（毫秒）
 * @return 接收到的所有 CAN 帧
 */
std::vector<can_frame> SocketCAN::receive_all_frames(int timeout_ms) {
    std::vector<can_frame> result_frames;
    if (sockfd_ == INIT_FD) {
        return result_frames;
    }

    // 1. 开启嗅探模式
    {
        std::lock_guard<std::mutex> lock(spy_mutex_);
        spy_buffer_.clear();
        is_spying_.store(true);
    }

    // 2. 临时放开过滤器，接收所有帧（注意：这可能会增加 CPU 负载）
    // 这一步是必要的，因为原本的过滤器可能过滤掉了我们要扫描的未知 ID
    can_filter filter{};
    filter.can_id = 0;
    filter.can_mask = 0;  // 接收所有帧
    setsockopt(sockfd_, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof(filter));

    // 3. 等待数据收集 (Sleep)
    // 我们不再这里 read，而是让 receiver_thread_ 帮我们读
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));

    // 4. 关闭嗅探模式
    is_spying_.store(false);

    // 5. 取出数据
    {
        std::lock_guard<std::mutex> lock(spy_mutex_);
        result_frames = spy_buffer_;  // 拷贝数据
        spy_buffer_.clear();
    }

    // 6. 恢复原来的过滤器
    {
        std::lock_guard<std::mutex> lock(can_callback_mutex_);
        update_can_filters_locked();  // 恢复只监听特定电机 ID
    }

    return result_frames;
}

void SocketCAN::update_can_filters_locked() {
    // 如果还没有打开 socket，则不设置过滤器
    if (sockfd_ == INIT_FD) {
        return;
    }

    can_filters_.clear();
    can_filters_.reserve(can_callback_list_.size());

    for (const auto &entry : can_callback_list_) {
        can_filter filter{};
        if (entry.second.is_extended) {
            // 扩展帧过滤器：匹配 bit 8-15 的 Motor ID。
            // RobStride 反馈帧格式通常为 [Type][Status][MotorID][MasterID]，
            // 让内核只把目标电机反馈帧送上来，避免用户态收到所有扩展帧。
            filter.can_id = ((entry.first & 0xFF) << 8) | CAN_EFF_FLAG;
            filter.can_mask = 0x0000FF00 | CAN_EFF_FLAG;
        } else {
            // 标准帧保持不变
            filter.can_id = entry.first & 0x7FF;
            filter.can_mask = 0x7FF;
        }
        can_filters_.push_back(filter);
    }

    // 如果没有任何过滤器，清空过滤规则（等价于接受所有帧）
    if (can_filters_.empty()) {
        setsockopt(sockfd_, SOL_CAN_RAW, CAN_RAW_FILTER, nullptr, 0);
    } else {
        setsockopt(sockfd_, SOL_CAN_RAW, CAN_RAW_FILTER,
                   can_filters_.data(),
                   static_cast<socklen_t>(can_filters_.size() * sizeof(can_filter)));
    }
}

void SocketCAN::receive() {
    // 接收逻辑已在receiver_thread_中实现
}
