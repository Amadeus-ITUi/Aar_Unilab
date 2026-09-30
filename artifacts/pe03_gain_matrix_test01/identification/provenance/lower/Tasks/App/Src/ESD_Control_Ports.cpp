#include "ESD_Control_Ports.h"

#include "CONF_ESD_Link_Task.h"
#include "NTFDCAN_Router.h"
#include "cmsis_os2.h"
#include "lib_adp_robstride.h"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace {

static_assert((OMX_ESD_CONTROL_RATE_HZ != 0u) &&
              ((1000u % OMX_ESD_CONTROL_RATE_HZ) == 0u),
              "ESD control rate must map to an integer millisecond period");

constexpr uint32_t kControlRateHz = OMX_ESD_CONTROL_RATE_HZ;
constexpr uint32_t kAdapterPeriodMs = 1000u / kControlRateHz;
constexpr uint16_t kSchemaCapabilities = (1u << 1) | (1u << 2);
constexpr uint32_t kQuiescentWaitMs = 20u;
constexpr float kPositionTargetMinRad = -FLT_MAX;
constexpr float kPositionTargetMaxRad = FLT_MAX;
constexpr float kSafeDampingNmSRad = 0.75f;
constexpr float kSafeDampingStopVelocityRadS = 0.02f;
constexpr uint16_t kSafeDampingDurationMs = 5000u;
constexpr uint16_t kWatchdogMs = 200u;

struct alignas(RobStrideMotor) motor_storage_t {
    std::byte bytes[sizeof(RobStrideMotor)];
};

static motor_storage_t s_motors[ESD_PORT_ACTIVE_MAX];
static esd_port_declaration_t s_declarations[ESD_PORT_ACTIVE_MAX];
static esd_port_declaration_t s_candidate[ESD_PORT_ACTIVE_MAX];
static esd_port_declaration_t s_backup[ESD_PORT_ACTIVE_MAX];
static esd_port_config_t s_ports[ESD_PORT_ACTIVE_MAX];
static can_channel_t s_channels[ESD_PORT_ACTIVE_MAX];
static uint8_t s_constructed[ESD_PORT_ACTIVE_MAX];
static size_t s_count;
static size_t s_constructed_count;
static size_t s_attached_count;
static bool s_initialized;

static can_channel_t channel_of(uint8_t logical_can)
{
    switch (logical_can) {
    case ESD_PORT_CAN1: return CAN_CH_NTFDCAN1;
    case ESD_PORT_CAN2: return CAN_CH_NTFDCAN2;
    case ESD_PORT_CAN3: return CAN_CH_NTFDCAN3;
    case ESD_PORT_CAN4: return CAN_CH_EXFDCAN0;
    default: return 0u;
    }
}

static bool is_robstride(uint8_t device)
{
    return device >= ESD_PORT_DEVICE_ROBSTRIDE_RS00 &&
           device <= ESD_PORT_DEVICE_ROBSTRIDE_RS06;
}

static esd_port_declaration_t normalize_declaration(
    const esd_port_declaration_t &source)
{
    esd_port_declaration_t result{};
    result.port_id = source.port_id;
    result.device = source.device;
    result.logical_can = source.logical_can;
    result.direction = 1;
    result.motor_id = source.port_id;
    result.master_id = 0xFDu;
    result.feedback_id = 0u;
    result.gear_ratio = 1.0f;
    result.motor_efficiency = 1.0f;
    result.position_min_rad = kPositionTargetMinRad;
    result.position_max_rad = kPositionTargetMaxRad;
    result.velocity_max_rad_s = source.velocity_max_rad_s;
    result.effort_max_nm = source.effort_max_nm;
    result.safe_damping_nm_s_rad = kSafeDampingNmSRad;
    result.safe_damping_stop_velocity_rad_s =
        kSafeDampingStopVelocityRadS;
    result.effort_to_backend = 1.0f;
    result.watchdog_ms = kWatchdogMs;
    result.safe_damping_duration_ms = kSafeDampingDurationMs;
    return result;
}

static bool valid_declaration(const esd_port_declaration_t &d)
{
    if (d.port_id == 0u || d.port_id >= ESD_PORT_ID_COUNT ||
        !is_robstride(d.device) ||
        channel_of(d.logical_can) == 0u || d.motor_id == 0u ||
        d.motor_id > 127u || d.master_id == 0u ||
        d.master_id == d.motor_id || d.feedback_id != 0u ||
        (d.direction != 1 && d.direction != -1) ||
        !std::isfinite(d.gear_ratio) || d.gear_ratio <= 0.0f ||
        !std::isfinite(d.motor_efficiency) || d.motor_efficiency <= 0.0f ||
        !std::isfinite(d.position_min_rad) ||
        !std::isfinite(d.position_max_rad) ||
        d.position_min_rad > d.position_max_rad ||
        !std::isfinite(d.velocity_max_rad_s) || d.velocity_max_rad_s <= 0.0f ||
        !std::isfinite(d.effort_max_nm) || d.effort_max_nm <= 0.0f ||
        !std::isfinite(d.safe_damping_nm_s_rad) ||
        d.safe_damping_nm_s_rad < 0.0f ||
        !std::isfinite(d.safe_damping_stop_velocity_rad_s) ||
        d.safe_damping_stop_velocity_rad_s < 0.0f ||
        d.safe_damping_stop_velocity_rad_s > d.velocity_max_rad_s ||
        !std::isfinite(d.effort_to_backend) || d.effort_to_backend <= 0.0f ||
        d.watchdog_ms < kAdapterPeriodMs ||
        d.safe_damping_duration_ms < 2u ||
        d.safe_damping_duration_ms > 5000u) {
        return false;
    }
    const auto model = static_cast<RobStrideModel>(
        d.device - ESD_PORT_DEVICE_ROBSTRIDE_RS00);
    const RobStrideProfile &profile = RobStride_GetProfile(model);
    const float motor_effort = d.effort_max_nm * d.effort_to_backend /
                               (d.gear_ratio * d.motor_efficiency);
    return std::isfinite(motor_effort) &&
           motor_effort <= profile.motion_torque_max_nm;
}

static void sort_declarations(esd_port_declaration_t *items, size_t count)
{
    for (size_t i = 1u; i < count; ++i) {
        const esd_port_declaration_t item = items[i];
        size_t j = i;
        while (j != 0u && item.port_id < items[j - 1u].port_id) {
            items[j] = items[j - 1u];
            --j;
        }
        items[j] = item;
    }
}

static bool validate_table(const esd_port_declaration_t *items, size_t count)
{
    if (items == nullptr || count == 0u || count > ESD_PORT_ACTIVE_MAX) {
        return false;
    }
    for (size_t i = 0u; i < count; ++i) {
        if (!valid_declaration(items[i])) return false;
        for (size_t j = 0u; j < i; ++j) {
            if (items[i].port_id == items[j].port_id) return false;
            if (items[i].logical_can == items[j].logical_can &&
                items[i].motor_id == items[j].motor_id) return false;
        }
    }
    return true;
}

static bool declaration_equal(const esd_port_declaration_t &a,
                              const esd_port_declaration_t &b)
{
    return a.port_id == b.port_id && a.device == b.device &&
           a.logical_can == b.logical_can && a.direction == b.direction &&
           a.motor_id == b.motor_id && a.master_id == b.master_id &&
           a.feedback_id == b.feedback_id && a.gear_ratio == b.gear_ratio &&
           a.motor_efficiency == b.motor_efficiency &&
           a.position_min_rad == b.position_min_rad &&
           a.position_max_rad == b.position_max_rad &&
           a.velocity_max_rad_s == b.velocity_max_rad_s &&
           a.effort_max_nm == b.effort_max_nm &&
           a.safe_damping_nm_s_rad == b.safe_damping_nm_s_rad &&
           a.safe_damping_stop_velocity_rad_s ==
               b.safe_damping_stop_velocity_rad_s &&
           a.effort_to_backend == b.effort_to_backend &&
           a.watchdog_ms == b.watchdog_ms &&
           a.safe_damping_duration_ms == b.safe_damping_duration_ms;
}

static bool same_table(const esd_port_declaration_t *items, size_t count)
{
    if (count != s_count) return false;
    for (size_t i = 0u; i < count; ++i) {
        if (!declaration_equal(items[i], s_declarations[i])) return false;
    }
    return true;
}

static bool zero_compatible(const esd_port_declaration_t &a,
                            const esd_port_declaration_t &b)
{
    return a.port_id == b.port_id && a.device == b.device &&
           a.logical_can == b.logical_can && a.direction == b.direction &&
           a.motor_id == b.motor_id && a.master_id == b.master_id &&
           a.feedback_id == b.feedback_id && a.gear_ratio == b.gear_ratio;
}

static bool construct_port(size_t index, const esd_port_declaration_t &d)
{
    esd_port_config_t &port = s_ports[index];
    uint8_t bus = 0u;
    uint8_t bus_index = 0u;
    if (!esd_port_resolve_can(d.logical_can, &bus, &bus_index)) return false;

    port = {};
    port.port_id = d.port_id;
    port.device = d.device;
    port.logical_can = d.logical_can;
    port.bus = bus;
    port.bus_index = bus_index;
    port.bus_address = d.motor_id;
    port.feedback_address = static_cast<uint16_t>(
        (static_cast<uint16_t>(d.motor_id) << 8) | d.master_id);
    port.controller_address = d.master_id;
    port.direction = d.direction;
    port.gear_ratio = d.gear_ratio;
    port.motor_efficiency = d.motor_efficiency;
    port.software_zero_rad = 0.0f;
    port.position_min_rad = d.position_min_rad;
    port.position_max_rad = d.position_max_rad;
    port.velocity_max_rad_s = d.velocity_max_rad_s;
    port.effort_max_nm = d.effort_max_nm;
    port.safe_damping_nm_s_rad = d.safe_damping_nm_s_rad;
    port.safe_damping_stop_velocity_rad_s =
        d.safe_damping_stop_velocity_rad_s;
    port.effort_to_backend = d.effort_to_backend;
    port.watchdog_ms = d.watchdog_ms;
    port.safe_damping_duration_ms = d.safe_damping_duration_ms;
    s_channels[index] = channel_of(d.logical_can);

    const auto model = static_cast<RobStrideModel>(
        d.device - ESD_PORT_DEVICE_ROBSTRIDE_RS00);
    auto *motor = new (s_motors[index].bytes)
        RobStrideMotor(model, d.motor_id, d.master_id);
    motor->set_offline_timeout(d.watchdog_ms);
    esd_port_bind_robstride(&port, motor);
    s_constructed[index] = 1u;
    ++s_constructed_count;
    return true;
}

static void destroy_port(size_t index)
{
    if (s_constructed[index] != 0u) {
        static_cast<RobStrideMotor *>(s_ports[index].user)->~RobStrideMotor();
        s_constructed[index] = 0u;
    }
}

static bool routers_quiescent(void)
{
    for (can_channel_t ch = CAN_CH_NTFDCAN1; ch <= CAN_CH_EXFDCAN0; ++ch) {
        if (!can_router_is_quiescent(ch)) return false;
    }
    return true;
}

static bool wait_routers_quiescent(void)
{
    for (uint32_t elapsed = 0u; elapsed <= kQuiescentWaitMs; ++elapsed) {
        if (routers_quiescent()) return true;
        if (elapsed != kQuiescentWaitMs) osDelay(1u);
    }
    return false;
}

static bool teardown_active(void)
{
    while (s_attached_count != 0u) {
        const size_t index = s_attached_count - 1u;
        auto *motor = static_cast<RobStrideMotor *>(s_ports[index].user);
        if (RobStride_DetachCan(s_channels[index], motor) != 0) return false;
        --s_attached_count;
    }
    if (!wait_routers_quiescent()) return false;
    while (s_constructed_count != 0u) destroy_port(--s_constructed_count);
    return true;
}

enum class build_result_t : uint8_t {
    ok,
    failed_clean,
    failed_dirty,
};

static build_result_t build_active(const esd_port_declaration_t *items,
                                   size_t count)
{
    std::memset(s_constructed, 0, sizeof(s_constructed));
    s_constructed_count = 0u;
    s_attached_count = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (!construct_port(i, items[i])) {
            return teardown_active() ? build_result_t::failed_clean
                                     : build_result_t::failed_dirty;
        }
    }
    esd_port_registry_t validation{};
    if (!esd_port_registry_init(&validation, s_ports, count,
                                kControlRateHz, kSchemaCapabilities)) {
        return teardown_active() ? build_result_t::failed_clean
                                 : build_result_t::failed_dirty;
    }
    for (size_t i = 0u; i < count; ++i) {
        auto *motor = static_cast<RobStrideMotor *>(s_ports[i].user);
        if (RobStride_AttachCan(s_channels[i], motor, kAdapterPeriodMs) != 0) {
            return teardown_active() ? build_result_t::failed_clean
                                     : build_result_t::failed_dirty;
        }
        ++s_attached_count;
    }
    return build_result_t::ok;
}

} // namespace

extern "C" bool ESD_ControlPorts_Initialize(esd_control_port_set_t *out)
{
    if (out == nullptr) return false;
    if (!s_initialized) {
        std::memset(s_declarations, 0, sizeof(s_declarations));
        std::memset(s_ports, 0, sizeof(s_ports));
        std::memset(s_constructed, 0, sizeof(s_constructed));
        s_count = 0u;
        s_constructed_count = 0u;
        s_attached_count = 0u;
        s_initialized = true;
    }
    out->ports = nullptr;
    out->count = 0u;
    return true;
}

extern "C" esd_control_ports_result_t ESD_ControlPorts_Reconfigure(
    const esd_port_declaration_t *declarations,
    size_t count,
    const float current_zero[ESD_PORT_ID_COUNT],
    esd_control_port_set_t *out,
    float next_zero[ESD_PORT_ID_COUNT])
{
    if (!s_initialized || declarations == nullptr || current_zero == nullptr ||
        out == nullptr || next_zero == nullptr || count == 0u ||
        count > ESD_PORT_ACTIVE_MAX) {
        return ESD_CONTROL_PORTS_INVALID;
    }
    for (size_t i = 0u; i < count; ++i) {
        s_candidate[i] = normalize_declaration(declarations[i]);
    }
    sort_declarations(s_candidate, count);
    if (!validate_table(s_candidate, count)) return ESD_CONTROL_PORTS_INVALID;

    if (same_table(s_candidate, count)) {
        std::memcpy(next_zero, current_zero,
                    ESD_PORT_ID_COUNT * sizeof(next_zero[0]));
        out->ports = s_count == 0u ? nullptr : s_ports;
        out->count = s_count;
        return ESD_CONTROL_PORTS_UNCHANGED;
    }

    std::memset(next_zero, 0, ESD_PORT_ID_COUNT * sizeof(next_zero[0]));
    for (size_t i = 0u; i < count; ++i) {
        for (size_t j = 0u; j < s_count; ++j) {
            if (zero_compatible(s_candidate[i], s_declarations[j])) {
                next_zero[s_candidate[i].port_id] =
                    current_zero[s_candidate[i].port_id];
                break;
            }
        }
    }

    const size_t old_count = s_count;
    std::memcpy(s_backup, s_declarations,
                old_count * sizeof(s_backup[0]));
    if (!teardown_active()) return ESD_CONTROL_PORTS_ROLLBACK_FAILED;

    const build_result_t candidate_result=build_active(s_candidate,count);
    if (candidate_result != build_result_t::ok) {
        if (candidate_result == build_result_t::failed_dirty) {
            s_count = 0u;
            out->ports = nullptr;
            out->count = 0u;
            return ESD_CONTROL_PORTS_ROLLBACK_FAILED;
        }
        if (old_count != 0u &&
            build_active(s_backup,old_count) != build_result_t::ok) {
            s_count = 0u;
            out->ports = nullptr;
            out->count = 0u;
            return ESD_CONTROL_PORTS_ROLLBACK_FAILED;
        }
        s_count = old_count;
        std::memcpy(s_declarations, s_backup,
                    old_count * sizeof(s_declarations[0]));
        out->ports = old_count == 0u ? nullptr : s_ports;
        out->count = old_count;
        return ESD_CONTROL_PORTS_ATTACH_FAILED;
    }

    s_count = count;
    std::memcpy(s_declarations, s_candidate,
                count * sizeof(s_declarations[0]));
    out->ports = s_ports;
    out->count = count;
    return ESD_CONTROL_PORTS_OK;
}

#ifdef ESD_CONTROL_PORTS_TESTING
extern "C" void ESD_ControlPorts_TestReset(void)
{
    (void)teardown_active();
    s_count = 0u;
    s_initialized = false;
}
#endif
