#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>

#include "motor_driver.hpp"
#include "SocketCAN.hpp"

enum DMError {
    DM_DOWN = 0x00,
    DM_UP = 0x01,
    OVER_VOLT = 0x08,
    UNDER_VOLT = 0x09,
    OVER_CURRENT = 0x0A,
    MOS_OVER_TEMP = 0x0B,
    COIL_OVER_TEMP = 0x0C,
    LOST_CONN = 0x0D,
    OVER_LOAD = 0x0E,
};

enum DM_Motor_Model { 
    DM4340P_48V, 
    DM10010L_48V,
    DM3507,
    DM3510,
    Num_Of_Motor 
};

enum DM_REG {
    UV_Value = 0,
    KT_Value = 1,
    OT_Value = 2,
    OC_Value = 3,
    ACC = 4,
    DEC = 5,
    MAX_SPD = 6,
    MST_ID = 7,
    ESC_ID = 8,
    TIMEOUT = 9,
    CTRL_MODE = 10,
    Damp = 11,
    Inertia = 12,
    hw_ver = 13,
    sw_ver = 14,
    SN = 15,
    NPP = 16,
    Rs = 17,
    LS = 18,
    Flux = 19,
    Gr = 20,
    PMAX = 21,
    VMAX = 22,
    TMAX = 23,
    I_BW = 24,
    KP_ASR = 25,
    KI_ASR = 26,
    KP_APR = 27,
    KI_APR = 28,
    OV_Value = 29,
    GREF = 30,
    Deta = 31,
    V_BW = 32,
    IQ_c1 = 33,
    VL_c1 = 34,
    can_br = 35,
    sub_ver = 36,
    u_off = 50,
    v_off = 51,
    k1 = 52,
    k2 = 53,
    m_off = 54,
    dir = 55,
    p_m = 80,
    xout = 81,
};

typedef struct {
    float PosMax;       ///< Maximum position limit (rad)
    float SpdMax;       ///< Maximum velocity limit (rad/s)
    float TauMax;       ///< Maximum torque limit (N·m)
    float OKpMax;       ///< Maximum outer-loop proportional gain
    float OKdMax;       ///< Maximum outer-loop derivative gain
} DM_Limit_Param;

class DmMotorDriver : public MotorDriver {
   public:
    DmMotorDriver(uint16_t motor_id, const std::string& interface_type, const std::string& can_interface, uint16_t master_id_offset,
                  DM_Motor_Model motor_model);
    ~DmMotorDriver();

    virtual void MotorLock() override;
    virtual void MotorUnlock() override;
    virtual uint8_t MotorInit() override;
    virtual void MotorDeInit() override;
    virtual bool MotorSetZero() override;
    virtual bool MotorWriteFlash() override;

    virtual void MotorGetParam(uint8_t param_cmd) override;
    virtual void MotorPosModeCmd(float pos, float spd, bool ignore_limit) override;
    virtual void MotorSpdModeCmd(float spd) override;
    virtual void MotorMitModeCmd(float kp, float kd, float q, float dq, float tau) override;
    virtual void MotorResetID() override {};
    virtual void set_motor_control_mode(uint8_t motor_control_mode) override;
    virtual int get_response_count() const {
        return response_count_;
    }
    virtual void refresh_motor_status() override;
    virtual void clear_motor_error() override;
    
    // 读取电机参数（带超时，用于在线检测）
    float read_motor_param_with_timeout(uint8_t param_cmd, int timeout_ms = 200);
    // 读取电机参数（带超时，支持指定 master_id）
    float read_motor_param_with_timeout(uint8_t param_cmd, int timeout_ms, uint16_t use_master_id);
    
    // 写入 master_id 并保存到 flash
    bool set_master_id(uint16_t new_master_id);

   private:
    std::atomic<int> response_count_{0};
    bool param_cmd_flag_[30] = {false};
    DM_Motor_Model motor_model_;
    DM_Limit_Param limit_param_;
    std::atomic<uint8_t> mos_temperature_{0};
    std::string can_interface_;
    std::shared_ptr<SocketCAN> can_;
    
    // 参数存储机制（用于参数读取）
    struct ParamValue {
        union {
            float floatValue;
            uint32_t uint32Value;
        } value;
        bool isFloat;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::unordered_map<uint8_t, ParamValue> param_map_;
    std::mutex param_map_mutex_;
    
    void set_motor_zero_dm();
    void clear_motor_error_dm();
    void write_register_dm(uint8_t rid, float value);
    void write_register_dm(uint8_t rid, int32_t value);
    void save_register_dm(uint8_t rid);
    void can_rx_cbk(const can_frame& rx_frame);
    
    // 辅助函数：判断寄存器是否为uint32类型
    static bool is_in_ranges(int number);
    static float uint8_to_float(const uint8_t data[4]);
};
