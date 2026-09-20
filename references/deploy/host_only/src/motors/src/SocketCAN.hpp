/**
 * @file SocketCAN.hpp
 * This file declares an interface to SocketCAN,
 * to facilitates frame transmission and reception.
 */

#pragma once

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <pthread.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <atomic>
#include <cstdbool>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <vector>

constexpr const int INIT_FD = -1;
constexpr const int TIMEOUT_SEC = 0;
constexpr const int TIMEOUT_USEC = 1000;

using CanCbkFunc = std::function<void(const can_frame &)>;
using CanCbkId = uint16_t;
// 回调信息：包含回调函数和帧类型（标准帧/扩展帧）
struct CanCallbackInfo {
    CanCbkFunc callback;
    bool is_extended;  // true=扩展帧, false=标准帧
};
using CanCbkMap = std::unordered_map<CanCbkId, CanCallbackInfo>;

class SocketCAN {
   private:
    std::string interface_;  // The network interface name
    int sockfd_ = -1;        // The file descriptor for the CAN socket
    std::atomic<bool> receiving_;

    sockaddr_can addr_;      // The address of the CAN socket
    ifreq if_request_;       // The network interface request

    /// Receiving
    std::thread receiver_thread_;
    CanCbkMap can_callback_list_;
    std::mutex can_callback_mutex_;
    // 当前已注册回调对应的 CAN 过滤器
    std::vector<can_filter> can_filters_;

    /// Spy mode for receive_all_frames (避免竞争)
    std::mutex spy_mutex_;
    std::vector<can_frame> spy_buffer_;
    std::atomic<bool> is_spying_{false};

    SocketCAN(std::string interface);

    static std::shared_ptr<SocketCAN> createInstance(const std::string &interface) {
        return std::shared_ptr<SocketCAN>(new SocketCAN(interface));
    }
    static std::shared_ptr<spdlog::logger> logger_;
    static std::unordered_map<std::string, std::shared_ptr<SocketCAN>> instances_;

   public:
    SocketCAN(const SocketCAN &) = delete;
    SocketCAN &operator=(const SocketCAN &) = delete;
    ~SocketCAN();
    static void init_logger(std::shared_ptr<spdlog::logger> logger) { logger_ = logger; }
    static std::shared_ptr<SocketCAN> get(std::string interface) {
        if (logger_.get() == nullptr) logger_ = spdlog::stdout_color_mt("SocketCAN");
        if (instances_.find(interface) == instances_.end()) instances_[interface] = createInstance(interface);
        return instances_[interface];
    }
    void open(std::string interface);
    void close();
    bool transmit(const can_frame &frame);
    void receive();
    void add_can_callback(const CanCbkFunc callback, const CanCbkId id, bool is_extended = true);
    void remove_can_callback(const CanCbkId id);
    void clear_can_callbacks();
    // 临时接收所有帧（用于扫描），返回接收到的帧列表
    std::vector<can_frame> receive_all_frames(int timeout_ms = 1000);

   private:
    // 根据当前回调列表刷新 CAN 过滤器，只接收关注的 can_id
    void update_can_filters_locked();
};
