#include "el05_motor_driver.hpp"
#include "timer.hpp"
#include <cstring>
#include <algorithm>
#include <iostream> // 记得添加头文件
#include <iomanip>
EL05MotorDriver::EL05MotorDriver(uint16_t motor_id, const std::string& can_interface, 
                                 uint16_t master_id)
    : MotorDriver(), can_interface_(can_interface), master_id_(master_id) {
    motor_id_ = motor_id;
    can_ = SocketCAN::get(can_interface);
    
    // 注册回调（私有协议使用扩展帧）
    auto callback = std::bind(&EL05MotorDriver::canRxCallback, this, 
                             std::placeholders::_1);
    can_->add_can_callback(callback, motor_id_, true);  // true = 扩展帧
}

EL05MotorDriver::~EL05MotorDriver() {
    can_->remove_can_callback(motor_id_);
}

uint32_t EL05MotorDriver::buildCanID(uint8_t comm_type, uint16_t data_field) const {
    uint32_t comm_type_bits = (comm_type & 0x1F) << 24;
    uint32_t data_field_bits = (data_field & 0xFFFF) << 8;
    uint32_t id_bits = motor_id_ & 0xFF;
    return comm_type_bits | data_field_bits | id_bits;
}

uint16_t EL05MotorDriver::floatToUint(float x, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    x = std::max(std::min(x, x_max), x_min);
    return static_cast<uint16_t>((x - offset) * ((1 << bits) - 1) / span);
}

float EL05MotorDriver::uintToFloat(uint16_t x_int, float x_min, float x_max, int bits) const {
    float span = x_max - x_min;
    float offset = x_min;
    uint16_t max_val = (1 << bits) - 1;
    if (max_val == 0) return offset;
    return static_cast<float>(x_int * span / max_val + offset);
}

void EL05MotorDriver::sendCommand(uint8_t comm_type, uint16_t data_field, const uint8_t* payload) {
    uint32_t real_can_id = buildCanID(comm_type, data_field);
    
    can_frame frame;
    frame.can_id = real_can_id | CAN_EFF_FLAG;  // 扩展帧
    frame.can_dlc = 8;
    std::memcpy(frame.data, payload, 8);
    
    if (can_->transmit(frame)) {
        response_count_++;
    }
}

void EL05MotorDriver::canRxCallback(const can_frame& rx_frame) {
    // 静态计数器，用于控制打印频率
    static int el05_print_count = 0;

    // 私有协议：必须是扩展帧
    if (!(rx_frame.can_id & CAN_EFF_FLAG)) {
        return;
    }
    
    uint32_t can_id = rx_frame.can_id & CAN_EFF_MASK;
    // 根据私有协议反馈帧格式：[Type(5位)][DataField(16位)][MotorID(8位)][MasterID(8位)]
    // 电机ID在 bit 8-15（反馈帧格式与发送帧不同，SocketCAN.cpp 中也是这么提取的）
    uint8_t motor_id = (can_id >> 8) & 0xFF;

    if (motor_id != motor_id_) return;
    
    // 收到反馈，清零计数器（修复：之前错误地使用了++）
    response_count_ = 0; 

    // --- 解析数据 (完全对齐 Python 逻辑) ---
    
    // 1. 位置、速度、力矩 (Big Endian)
    uint16_t pos_uint = (rx_frame.data[0] << 8) | rx_frame.data[1];
    uint16_t vel_uint = (rx_frame.data[2] << 8) | rx_frame.data[3];
    uint16_t torque_uint = (rx_frame.data[4] << 8) | rx_frame.data[5];
    
    // 2. [修正] 温度是 Byte 6 和 Byte 7 组成的 uint16 (Python: int.from_bytes(data[6:8]))
    uint16_t temp_raw = (rx_frame.data[6] << 8) | rx_frame.data[7];
    
    // 3. [修正] 故障信息来自 CAN ID 的 data_field 高 6 位
    // Python: fault_info = (data_field >> 8) & 0b111111
    // CAN ID 结构: [Type(5)][DataField(16)][MotorID(8)]
    // DataField 位于 bit 8-23。 (can_id >> 8) & 0xFFFF 得到 DataField
    uint16_t data_field = (can_id >> 8) & 0xFFFF;
    uint8_t fault_info = (data_field >> 8) & 0x3F;
    
    // --- 转换物理量 ---
    motor_pos_ = uintToFloat(pos_uint, P_MIN, P_MAX, 16);
    motor_spd_ = uintToFloat(vel_uint, V_MIN, V_MAX, 16);
    motor_current_ = uintToFloat(torque_uint, T_MIN, T_MAX, 16);
    
    // [修正] 温度计算：原始值除以 10.0
    motor_temperature_ = static_cast<float>(temp_raw) / 10.0f;
    
    // 更新错误码 (将 fault_info 存入 error_id_)
    error_id_ = fault_info;
    if (el05_print_count <= 5) {
        printf("[EL05 Motor %d] Pos: %.4f | Vel: %.4f | Torque: %.4f | Temp: %.1f C | Fault: 0x%02X\n", 
                motor_id_, 
                motor_pos_.load(), 
                motor_spd_.load(), 
                motor_current_.load(), 
                motor_temperature_.load(),
                error_id_.load());
    }
    el05_print_count++;
}


void EL05MotorDriver::MotorLock() {
    // 私有协议：disable命令（通信类型4）
    uint8_t payload[8] = {0};
    sendCommand(0x04, master_id_, payload);
}

void EL05MotorDriver::MotorUnlock() {
    // 私有协议：enable命令（通信类型3）
    uint8_t payload[8] = {0};
    sendCommand(0x03, master_id_, payload);
}

uint8_t EL05MotorDriver::MotorInit() {
    MotorUnlock();
    Timer::ThreadSleepFor(100);
    set_motor_control_mode(0);  // 0 = 运控模式（move_control_mode）
    Timer::ThreadSleepFor(100);
    refresh_motor_status();
    Timer::ThreadSleepFor(100);
    return error_id_;
}

void EL05MotorDriver::MotorDeInit() {
    MotorLock();
    Timer::ThreadSleepFor(100);
}

bool EL05MotorDriver::MotorSetZero() {
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

bool EL05MotorDriver::MotorWriteFlash() {
    uint8_t payload[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    sendCommand(0x16, master_id_, payload);
    return true;
}

void EL05MotorDriver::MotorGetParam(uint8_t param_cmd) {
    // 实现参数读取
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);  // 请求反馈
}

void EL05MotorDriver::MotorPosModeCmd(float pos, float spd, bool ignore_limit) {
    // EL05主要使用MIT模式，位置模式需要设置参数
    set_motor_control_mode(POS);
    // 实现位置控制命令
}

void EL05MotorDriver::MotorSpdModeCmd(float spd) {
    set_motor_control_mode(SPD);
    // 实现速度控制命令
}

void EL05MotorDriver::MotorMitModeCmd(float f_p, float f_v, float f_kp, float f_kd, float f_t) {
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
    // CAN ID格式：通信类型(5位) | 力矩(16位) | 主机ID(8位) | 电机ID(8位)
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

void EL05MotorDriver::set_motor_control_mode(uint8_t motor_control_mode) {
    motor_control_mode_ = motor_control_mode;
    // 私有协议：设置运行模式（参数0x7005）
    // 0=运控模式, 1=位置模式, 2=速度模式, 3=电流模式, 4=零点模式, 5=位置模式CSP
    uint8_t payload[8] = {0};
    payload[0] = 0x05;  // 参数地址低字节
    payload[1] = 0x70;  // 参数地址高字节
    payload[4] = static_cast<uint8_t>(motor_control_mode);
    sendCommand(0x12, master_id_, payload);  // 通信类型0x12 = 设置参数
}

void EL05MotorDriver::refresh_motor_status() {
    // 私有协议：请求电机反馈（通信类型2）
    uint8_t payload[8] = {0};
    sendCommand(0x02, master_id_, payload);
}

void EL05MotorDriver::clear_motor_error() {
    // 私有协议：disable然后enable
    uint8_t payload[8] = {0};
    sendCommand(0x04, master_id_, payload);
    Timer::ThreadSleepFor(100);
    sendCommand(0x03, master_id_, payload);
}

void EL05MotorDriver::MotorResetID() {
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
