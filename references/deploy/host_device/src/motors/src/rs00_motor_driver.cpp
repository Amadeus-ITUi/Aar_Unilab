#include "rs00_motor_driver.hpp"
#include "timer.hpp"
#include <cstring>
#include <algorithm>
#include <map>
#include <cstdio>
#include <iostream> // 记得添加头文件
#include <iomanip>
RS00MotorDriver::RS00MotorDriver(uint16_t motor_id, const std::string& can_interface, 
                                 uint16_t master_id)
    : MotorDriver(), can_interface_(can_interface), master_id_(master_id) {
    motor_id_ = motor_id;
    can_ = SocketCAN::get(can_interface);
    
    // 注册回调（私有协议使用扩展帧）
    auto callback = std::bind(&RS00MotorDriver::canRxCallback, this, 
                             std::placeholders::_1);
    can_->add_can_callback(callback, motor_id_, true);  // true = 扩展帧
}

RS00MotorDriver::~RS00MotorDriver() {
    can_->remove_can_callback(motor_id_);
}

uint32_t RS00MotorDriver::buildCanID(uint8_t comm_type, uint16_t data_field) const {
    uint32_t comm_type_bits = (comm_type & 0x1F) << 24;
    uint32_t data_field_bits = (data_field & 0xFFFF) << 8;
    uint32_t id_bits = motor_id_ & 0xFF;
    return comm_type_bits | data_field_bits | id_bits;
}

uint16_t RS00MotorDriver::floatToUint(float x, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    x = std::max(std::min(x, x_max), x_min);
    return static_cast<uint16_t>((x - offset) * ((1 << bits) - 1) / span);
}

float RS00MotorDriver::uintToFloat(uint16_t x_int, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    uint16_t max_val = (1 << bits) - 1;
    if (max_val == 0) return offset;
    return static_cast<float>(x_int * span / max_val + offset);
}

// 参考代码的转换公式：((value / 32767.0f) - 1.0f) * range
// 这个公式将 0-65535 映射到 -range 到 +range
float RS00MotorDriver::referenceCodeConvert(uint16_t value, float range) const {
    return ((static_cast<float>(value) / 32767.0f) - 1.0f) * range;
}

void RS00MotorDriver::sendCommand(uint8_t comm_type, uint16_t data_field, const uint8_t* payload) {
    uint32_t real_can_id = buildCanID(comm_type, data_field);
    
    can_frame frame;
    frame.can_id = real_can_id | CAN_EFF_FLAG;  // 扩展帧
    frame.can_dlc = 8;
    std::memcpy(frame.data, payload, 8);
    
    if (can_->transmit(frame)) {
        response_count_++;
    }
}

void RS00MotorDriver::canRxCallback(const can_frame& rx_frame) {
    // 私有协议：扩展帧
    if (!(rx_frame.can_id & CAN_EFF_FLAG)) {
        return;
    }
   
    uint32_t can_id = rx_frame.can_id & CAN_EFF_MASK;
    uint8_t comm_type = (can_id >> 24) & 0x1F;
    // 私有协议反馈帧：[Type][Status/Fault][MotorID][MasterID]
    // MotorID 位于 bit 8-15；不兼容低 8 位命令帧布局，避免误解析。
    uint8_t motor_id = (can_id >> 8) & 0xFF;
    // 调试：打印所有接收到的CAN帧（仅前几次，避免刷屏）
    static thread_local std::map<uint16_t, int> debug_frame_count_map;
    int& debug_count = debug_frame_count_map[motor_id_];
    // if (debug_count < 5) {
    //     printf("[RS00 Motor %d] Received CAN frame: ID=0x%08X, comm_type=0x%02X, motor_id=%d, data=[%02X %02X %02X %02X %02X %02X %02X %02X]\n",
    //            motor_id_, can_id, comm_type, motor_id,
    //            rx_frame.data[0], rx_frame.data[1], rx_frame.data[2], rx_frame.data[3],
    //            rx_frame.data[4], rx_frame.data[5], rx_frame.data[6], rx_frame.data[7]);
    //     debug_count++;
    // }
    
    // 只处理电机反馈帧（通信类型0x02）
    if (motor_id != motor_id_ || comm_type != 0x02) {
        // 过滤掉非运动数据帧（如故障反馈帧0x15等）
        // 调试信息可以保留用于排查问题
        if (debug_count <= 3) {
            printf("[RS00 Motor %d] Frame filtered: motor_id match=%d (expected %d, got %d), comm_type match=%d (expected 0x02, got 0x%02X)\n",
                   motor_id_, (motor_id == motor_id_ ? 1 : 0), motor_id_, motor_id,
                   (comm_type == 0x02 ? 1 : 0), comm_type);
            debug_count++;
        }
        return;
    }
    
    response_count_ = 0;
    frame_counter_++;
    
    // 解析反馈数据（8字节）- 根据参考代码和说明书格式
    // 高字节在前（大端序）
    uint16_t pos_uint = (rx_frame.data[0] << 8) | rx_frame.data[1];
    uint16_t vel_uint = (rx_frame.data[2] << 8) | rx_frame.data[3];
    uint16_t torque_uint = (rx_frame.data[4] << 8) | rx_frame.data[5];
    uint16_t temperature_uint = (rx_frame.data[6] << 8) | rx_frame.data[7];  // 修复：温度是16位
    
    // 使用标准的线性映射转换，与RS05保持一致
    // 参考代码中，RS00的参数范围：position=4*PI, velocity=33, torque=14

    // 解析并验证位置数据
    float pos_raw = uintToFloat(pos_uint, P_MIN, P_MAX, 16);
    if (pos_raw >= P_MIN - 1.0f && pos_raw <= P_MAX + 1.0f) {  // 允许一定的误差范围
        motor_pos_ = pos_raw;
    } else {
        static thread_local int pos_warn_count = 0;
        if (pos_warn_count < 3) {
            printf("[RS00 Motor %d] Invalid position: %.3f rad (raw: %u), keeping previous value\n",
                   motor_id_, pos_raw, pos_uint);
            pos_warn_count++;
        }
    }

    // 解析并验证速度数据
    float spd_raw = uintToFloat(vel_uint, V_MIN, V_MAX, 16);
    if (spd_raw >= V_MIN - 5.0f && spd_raw <= V_MAX + 5.0f) {  // 允许一定的误差范围
        motor_spd_ = spd_raw;
    } else {
        static thread_local int spd_warn_count = 0;
        if (spd_warn_count < 3) {
            printf("[RS00 Motor %d] Invalid velocity: %.3f rad/s (raw: %u), keeping previous value\n",
                   motor_id_, spd_raw, vel_uint);
            spd_warn_count++;
        }
    }

    // 解析并验证力矩数据
    float torque_raw = uintToFloat(torque_uint, T_MIN, T_MAX, 16);
    if (torque_raw >= T_MIN - 2.0f && torque_raw <= T_MAX + 2.0f) {  // 允许一定的误差范围
        motor_current_ = torque_raw;
    } else {
        static thread_local int torque_warn_count = 0;
        if (torque_warn_count < 3) {
            printf("[RS00 Motor %d] Invalid torque: %.3f Nm (raw: %u), keeping previous value\n",
                   motor_id_, torque_raw, torque_uint);
            torque_warn_count++;
        }
    }

    // 温度解析：添加数据验证，过滤异常值
    float temp_raw = static_cast<float>(temperature_uint) * 0.1f;  // 温度单位：0.1°C
    if (temp_raw >= 0.0f && temp_raw <= 150.0f) {  // 合理的温度范围：0-150°C
        motor_temperature_ = temp_raw;
    } else {
        // 温度数据异常，保持上次有效值，但只在调试模式下打印警告
        static thread_local int temp_warn_count = 0;
        if (temp_warn_count < 3) {
            printf("[RS00 Motor %d] Invalid temperature: %.1f°C (raw: %u), keeping previous value\n",
                   motor_id_, temp_raw, temperature_uint);
            temp_warn_count++;
        }
    }
    
    // 错误码：根据说明书，Byte7的低位可能包含错误信息
    // 但参考代码中没有明确解析error字段，这里保持原样
    error_id_ = 0;  // 如果需要解析错误，需要查看说明书的错误位定义
    
    // 每10帧打印一次反馈信息（便于调试，可以改为100）
    int current_frame = frame_counter_.load();
    if ( current_frame <= 5) {
        printf("[Motor %d] Frame #%d - Pos: %.4f rad, Vel: %.4f rad/s, Torque: %.4f Nm, Temp: %.1f°C\n",
               motor_id_, current_frame, 
               motor_pos_.load(), motor_spd_.load(), motor_current_.load(), motor_temperature_.load());
    }
}


void RS00MotorDriver::MotorLock() {
    // 私有协议：disable命令（通信类型4）
    uint8_t payload[8] = {0};
    // 失能是安全关键的状态切换命令。重复发送，降低单帧丢失后
    // 驱动器继续保持最后一条 MIT/PD 指令的风险。
    for (int i = 0; i < 10; ++i) {
        sendCommand(0x04, master_id_, payload);
        if (i < 9) {
            Timer::ThreadSleepFor(10);
        }
    }
}

void RS00MotorDriver::MotorUnlock() {
    // 私有协议：enable命令（通信类型3）
    uint8_t payload[8] = {0};
    // 使能是状态切换命令，连续发送几次可以降低偶发丢帧导致未使能的概率。
    for (int i = 0; i < 5; ++i) {
        sendCommand(0x03, master_id_, payload);
        Timer::ThreadSleepFor(10);
    }
    Timer::ThreadSleepFor(500);
}

uint8_t RS00MotorDriver::MotorInit() {
    MotorUnlock();
    Timer::ThreadSleepFor(100);
    set_motor_control_mode(0);  // 0 = 运控模式（move_control_mode）
    Timer::ThreadSleepFor(100);
    refresh_motor_status();
    Timer::ThreadSleepFor(100);
    return error_id_;
}

void RS00MotorDriver::MotorDeInit() {
    MotorLock();
    Timer::ThreadSleepFor(100);
}

bool RS00MotorDriver::MotorSetZero() {
    // 私有协议：设置零点流程
    // 参考代码：先失能电机，切换到速度模式，然后发送归零命令
    
    // 步骤1: 失能电机（安全措施）
    MotorLock();
    Timer::ThreadSleepFor(100);
    
    // 步骤2: 切换到速度模式（模式2）
    // 注意：参考代码在Set_ZeroPos中使用速度模式，而不是零点模式
    // 零点模式（模式4）只是让电机进入零点设置状态，但实际归零命令需要在速度模式下执行
    set_motor_control_mode(2);  // 速度模式
    Timer::ThreadSleepFor(100);
    
    // 步骤3: 发送归零命令（通信类型6）
    uint8_t payload[8] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    sendCommand(0x06, master_id_, payload);
    Timer::ThreadSleepFor(500);  // 等待归零完成
    
    // 步骤4: 使能电机（如果需要继续使用）
    // 注意：这里不自动使能，由调用者决定是否使能
    // MotorUnlock();
    
    return true;
}

bool RS00MotorDriver::MotorWriteFlash() {
    uint8_t payload[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    sendCommand(0x16, master_id_, payload);
    return true;
}

void RS00MotorDriver::MotorGetParam(uint8_t param_cmd) {
    // 实现参数读取
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);  // 请求反馈
}

void RS00MotorDriver::MotorPosModeCmd(float pos, float spd, bool ignore_limit) {
    // RS00主要使用MIT模式，位置模式需要设置参数
    set_motor_control_mode(POS);
    // 实现位置控制命令
}

void RS00MotorDriver::MotorSpdModeCmd(float spd) {
    set_motor_control_mode(SPD);
    // 实现速度控制命令
}

void RS00MotorDriver::MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
    // 安全检查：过滤 NaN/Inf
    if (!std::isfinite(f_p) || !std::isfinite(f_v) ||
        !std::isfinite(f_kp) || !std::isfinite(f_kd) || !std::isfinite(f_t)) {
        return;
    }
    // 硬限幅，防止超出物理能力
    f_p = std::clamp(f_p, P_MIN, P_MAX);
    f_v = std::clamp(f_v, V_MIN, V_MAX);
    f_kp = std::clamp(f_kp, KP_MIN, KP_MAX);
    f_kd = std::clamp(f_kd, KD_MIN, KD_MAX);
    f_t = std::clamp(f_t, T_MIN, T_MAX);

    // 私有协议：运控模式（通信类型0x01）
    // 数据格式：位置(16位) + 速度(16位) + Kp(16位) + Kd(16位)
    // CAN ID格式：通信类型(5位) | 力矩(16位) | 电机ID(8位)
    uint16_t t_uint = floatToUint(f_t, T_MIN, T_MAX, 16);
    uint16_t p_uint = floatToUint(f_p, P_MIN, P_MAX, 16);
    uint16_t v_uint = floatToUint(f_v, V_MIN, V_MAX, 16);
    uint16_t kp_uint = floatToUint(f_kp, KP_MIN, KP_MAX, 16);
    uint16_t kd_uint = floatToUint(f_kd, KD_MIN, KD_MAX, 16);
    
    uint8_t payload[8];
    payload[0] = (p_uint >> 8) & 0xFF;  // 位置高字节
    payload[1] = p_uint & 0xFF;          // 位置低字节
    payload[2] = (v_uint >> 8) & 0xFF;  // 速度高字节
    payload[3] = v_uint & 0xFF;          // 速度低字节
    payload[4] = (kp_uint >> 8) & 0xFF;  // Kp高字节
    payload[5] = kp_uint & 0xFF;         // Kp低字节
    payload[6] = (kd_uint >> 8) & 0xFF;  // Kd高字节
    payload[7] = kd_uint & 0xFF;         // Kd低字节
    
    // 通信类型0x01（运控模式），数据字段为力矩值
    sendCommand(0x01, t_uint, payload);
}

void RS00MotorDriver::set_motor_control_mode(uint8_t motor_control_mode) {
    motor_control_mode_ = motor_control_mode;
    // 私有协议：设置运行模式（参数0x7005）
    // 0=运控模式, 1=位置模式, 2=速度模式, 3=电流模式, 4=零点模式, 5=位置模式CSP
    uint8_t payload[8] = {0};
    payload[0] = 0x05;  // 参数地址低字节
    payload[1] = 0x70;  // 参数地址高字节
    payload[4] = static_cast<uint8_t>(motor_control_mode);
    sendCommand(0x12, master_id_, payload);  // 通信类型0x12 = 设置参数
}

void RS00MotorDriver::refresh_motor_status() {
    // 私有协议：请求电机反馈（通信类型2）
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);
}

void RS00MotorDriver::clear_motor_error() {
    // 私有协议：disable然后enable
    uint8_t payload[8] = {0};
    sendCommand(0x04, master_id_, payload);
    Timer::ThreadSleepFor(100);
    sendCommand(0x03, master_id_, payload);
}

void RS00MotorDriver::MotorResetID() {
    // 识别电机ID：发送查询命令到广播ID，等待电机回复
    // 电机上电时会发送广播帧 0x7FFE，我们发送查询命令让它回复实际ID
    
    if (logger_) {
        logger_->info("[Motor {}] Querying motor ID from broadcast address...", motor_id_);
    }
    
    // 发送查询命令到广播ID 0x7FFE（私有协议）
    // 通信类型 0x11 (参数读取), 参数地址 0x7001 (电机ID参数)
    uint32_t comm_type_bits = (0x11 & 0x1F) << 24;  // 通信类型 0x11 (读取参数)
    uint32_t data_field_bits = (0x7FFE & 0xFFFF) << 8;  // 数据字段使用广播ID
    uint32_t id_bits = 0xFE & 0xFF;  // 广播ID的低8位是 0xFE
    uint32_t broadcast_can_id = comm_type_bits | data_field_bits | id_bits;
    
    can_frame query_frame{};
    query_frame.can_id = broadcast_can_id | CAN_EFF_FLAG;
    query_frame.can_dlc = 8;
    
    // 参数读取格式: [参数地址低字节, 参数地址高字节, ...]
    query_frame.data[0] = 0x01;  // 参数地址低字节 (0x7001 - 电机ID)
    query_frame.data[1] = 0x70;  // 参数地址高字节 (0x7001)
    std::memset(&query_frame.data[2], 0, 6);
    
    can_->transmit(query_frame);
    Timer::ThreadSleepFor(200);
    
    if (logger_) {
        logger_->info("[Motor {}] ID query command sent to broadcast address 0x7FFE", motor_id_);
        logger_->info("[Motor {}] Waiting for motor response... (check CAN bus for reply frames)", motor_id_);
        logger_->info("[Motor {}] Motor will reply with its actual ID in the CAN frame ID (low 8 bits)", motor_id_);
    }
}
