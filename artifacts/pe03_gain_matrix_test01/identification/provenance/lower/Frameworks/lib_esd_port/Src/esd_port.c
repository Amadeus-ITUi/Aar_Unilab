#include "esd_port.h"

#include "esd_link_v1_wire.h"

#include <math.h>
#include <string.h>

static float clampf(float value, float low, float high)
{
    return (value < low) ? low : ((value > high) ? high : value);
}

bool esd_port_resolve_can(uint8_t logical_can, uint8_t *bus,
                          uint8_t *bus_index)
{
    if ((bus == NULL) || (bus_index == NULL)) {
        return false;
    }
    switch (logical_can) {
    case ESD_PORT_CAN1:
    case ESD_PORT_CAN2:
    case ESD_PORT_CAN3:
        *bus = ESD_PORT_BUS_NTFDCAN;
        *bus_index = logical_can;
        return true;
    case ESD_PORT_CAN4:
        *bus = ESD_PORT_BUS_EXFDCAN;
        *bus_index = 0u;
        return true;
    default:
        return false;
    }
}

typedef struct {
    uint32_t rx_id;
    uint32_t tx_id;
    uint8_t extended;
    uint8_t grouped;
    uint8_t group_slot;
} port_route_t;

static bool resolve_route(const esd_port_config_t *port, port_route_t *route)
{
    if ((port == NULL) || (route == NULL)) {
        return false;
    }
    route->rx_id = port->feedback_address;
    route->extended = 0u;
    route->grouped = 0u;
    route->group_slot = 0u;
    switch (port->device) {
    case ESD_PORT_DEVICE_DM4310:
    case ESD_PORT_DEVICE_DM4340:
        if ((port->bus_address > 255u) ||
            (port->feedback_address > 0x7FFu) ||
            (port->controller_address != 0u)) return false;
        route->tx_id = port->bus_address;
        return true;
    case ESD_PORT_DEVICE_LK8016E:
        if ((port->bus_address < 1u) || (port->bus_address > 32u) ||
            (port->feedback_address > 0x7FFu) ||
            (port->controller_address != 0u)) return false;
        route->tx_id = (uint16_t)(0x140u + port->bus_address);
        return route->rx_id == route->tx_id;
    case ESD_PORT_DEVICE_DJI_C610:
    case ESD_PORT_DEVICE_DJI_C620:
        if ((port->bus_address < 1u) || (port->bus_address > 8u) ||
            (port->feedback_address > 0x7FFu) ||
            (port->controller_address != 0u)) return false;
        route->tx_id = (port->bus_address <= 4u) ? 0x200u : 0x1FFu;
        route->grouped = 1u;
        route->group_slot = (uint8_t)((port->bus_address - 1u) & 3u);
        return route->rx_id <= 0x7FFu;
    case ESD_PORT_DEVICE_DJI_GM6020:
        if ((port->bus_address < 1u) || (port->bus_address > 7u) ||
            (port->feedback_address > 0x7FFu) ||
            (port->controller_address != 0u)) return false;
        route->tx_id = (port->bus_address <= 4u) ? 0x1FFu : 0x2FFu;
        route->grouped = 1u;
        route->group_slot = (uint8_t)((port->bus_address - 1u) & 3u);
        return route->rx_id <= 0x7FFu;
    case ESD_PORT_DEVICE_ROBSTRIDE_RS00:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS01:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS02:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS03:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS04:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS05:
    case ESD_PORT_DEVICE_ROBSTRIDE_RS06:
        if ((port->bus_address < 1u) || (port->bus_address > 127u) ||
            (port->controller_address < 1u) ||
            (port->controller_address > 255u) ||
            (port->controller_address == port->bus_address)) return false;
        route->extended = 1u;
        route->rx_id = ((uint32_t)port->bus_address << 8) |
                       port->controller_address;
        route->tx_id = port->bus_address;
        return port->feedback_address == route->rx_id;
    default:
        return false;
    }
}

static bool valid_port(const esd_port_config_t *port)
{
    uint8_t expected_bus = 0u;
    uint8_t expected_index = 0u;
    port_route_t route;
    return (port != NULL) && (port->port_id < ESD_PORT_ID_COUNT) &&
           esd_port_resolve_can(port->logical_can, &expected_bus,
                                &expected_index) &&
           (port->bus == expected_bus) && (port->bus_index == expected_index) &&
           resolve_route(port, &route) &&
           ((port->direction == 1) || (port->direction == -1)) &&
           isfinite(port->gear_ratio) && (port->gear_ratio > 0.0f) &&
           isfinite(port->motor_efficiency) &&
           (port->motor_efficiency > 0.0f) &&
           isfinite(port->software_zero_rad) &&
           isfinite(port->position_min_rad) &&
           isfinite(port->position_max_rad) &&
           (port->position_min_rad <= port->position_max_rad) &&
           isfinite(port->velocity_max_rad_s) &&
           (port->velocity_max_rad_s > 0.0f) &&
           isfinite(port->effort_max_nm) && (port->effort_max_nm > 0.0f) &&
           isfinite(port->safe_damping_nm_s_rad) &&
           (port->safe_damping_nm_s_rad >= 0.0f) &&
           isfinite(port->safe_damping_stop_velocity_rad_s) &&
           (port->safe_damping_stop_velocity_rad_s >= 0.0f) &&
           (port->safe_damping_stop_velocity_rad_s <=
            port->velocity_max_rad_s) &&
           isfinite(port->effort_to_backend) &&
           (port->effort_to_backend > 0.0f) &&
           (port->watchdog_ms >= 2u) &&
           (port->safe_damping_duration_ms >= 2u) &&
           (port->safe_damping_duration_ms <= 5000u) &&
           (port->ops != NULL) &&
           (port->ops->sample != NULL) &&
           (port->ops->set_motor_effort != NULL) &&
           (port->ops->request_enable != NULL) &&
           (port->ops->emergency_disable != NULL) &&
           (port->ops->verify_enabled != NULL) &&
           (port->ops->verify_disabled != NULL);
}

static bool routes_conflict(const esd_port_config_t *lhs,
                            const esd_port_config_t *rhs)
{
    if (lhs->logical_can != rhs->logical_can) {
        return false;
    }
    port_route_t a;
    port_route_t b;
    if (!resolve_route(lhs, &a) || !resolve_route(rhs, &b)) {
        return true;
    }
    if (a.extended != b.extended) {
        return false;
    }
    if (a.rx_id == b.rx_id) {
        return true;
    }
    if (a.tx_id != b.tx_id) {
        return false;
    }
    if (!a.grouped || !b.grouped) {
        return true;
    }
    return a.group_slot == b.group_slot;
}

bool esd_port_registry_init(esd_port_registry_t *registry,
                            const esd_port_config_t *ports,
                            size_t count,
                            uint32_t control_rate_hz,
                            uint16_t schema_capabilities)
{
    if ((registry == NULL) || (count > ESD_PORT_ACTIVE_MAX) ||
        ((count != 0u) && (ports == NULL)) ||
        (control_rate_hz == 0u) || (control_rate_hz > 1000u) ||
        ((1000u % control_rate_hz) != 0u)) {
        return false;
    }
    const uint32_t control_period_ms = 1000u / control_rate_hz;
    uint32_t used = 0u;
    for (size_t i = 0u; i < count; ++i) {
        if (!valid_port(&ports[i])) {
            return false;
        }
        if (ports[i].watchdog_ms < control_period_ms) {
            return false;
        }
        const uint32_t bit = 1u << ports[i].port_id;
        if ((used & bit) != 0u) {
            return false;
        }
        for (size_t j = 0u; j < i; ++j) {
            if (routes_conflict(&ports[j], &ports[i])) {
                return false;
            }
        }
        used |= bit;
    }
    registry->ports = ports;
    registry->count = count;
    registry->control_rate_hz = control_rate_hz;
    registry->schema_capabilities = schema_capabilities;
    registry->fingerprint = esd_port_compute_fingerprint(registry);
    return true;
}

const esd_port_config_t *esd_port_find(const esd_port_registry_t *registry,
                                       uint8_t port_id)
{
    if (registry != NULL) {
        for (size_t i = 0u; i < registry->count; ++i) {
            if (registry->ports[i].port_id == port_id) {
                return &registry->ports[i];
            }
        }
    }
    return NULL;
}

uint32_t esd_port_compute_fingerprint(const esd_port_registry_t *registry)
{
    return esd_port_compute_fingerprint_with_offsets(registry, NULL);
}

static void crc_u8(uint32_t *crc, uint8_t value)
{
    *crc = esd_link_v1_crc32_extend(*crc, &value, sizeof(value));
}

static void crc_u16(uint32_t *crc, uint16_t value)
{
    uint8_t bytes[2];
    esd_link_v1_le16_write(bytes, value);
    *crc = esd_link_v1_crc32_extend(*crc, bytes, sizeof(bytes));
}

static void crc_u32(uint32_t *crc, uint32_t value)
{
    uint8_t bytes[4];
    esd_link_v1_le32_write(bytes, value);
    *crc = esd_link_v1_crc32_extend(*crc, bytes, sizeof(bytes));
}

static void crc_f32(uint32_t *crc, float value)
{
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    crc_u32(crc, bits);
}

uint32_t esd_port_compute_fingerprint_with_offsets(
    const esd_port_registry_t *registry,
    const float offsets[ESD_PORT_ID_COUNT])
{
    if (registry == NULL) {
        return 0u;
    }
    uint32_t crc = esd_link_v1_crc32_begin();
    uint32_t mask = 0u;
    for (size_t i = 0u; i < registry->count; ++i) {
        mask |= 1u << registry->ports[i].port_id;
    }
    crc_u32(&crc, mask);
    crc_u32(&crc, registry->control_rate_hz);
    crc_u16(&crc, registry->schema_capabilities);
    crc_u8(&crc, (uint8_t)registry->count);
    crc_u8(&crc, 0u);

    /* Port id order and explicit widths make this independent of ABI padding. */
    for (uint8_t id = 0u; id < ESD_PORT_ID_COUNT; ++id) {
        const esd_port_config_t *p = esd_port_find(registry, id);
        if (p == NULL) {
            continue;
        }
        crc_u8(&crc, p->port_id);
        crc_u8(&crc, p->device);
        crc_u8(&crc, p->logical_can);
        crc_u8(&crc, p->bus);
        crc_u8(&crc, p->bus_index);
        crc_u16(&crc, p->bus_address);
        crc_u16(&crc, p->feedback_address);
        crc_u16(&crc, p->controller_address);
        crc_u8(&crc, (uint8_t)p->direction);
        crc_u8(&crc, p->confirm_mode);
        crc_f32(&crc, p->gear_ratio);
        crc_f32(&crc, p->motor_efficiency);
        crc_f32(&crc, (offsets != NULL) ? offsets[id]
                                        : p->software_zero_rad);
        crc_f32(&crc, p->position_min_rad);
        crc_f32(&crc, p->position_max_rad);
        crc_f32(&crc, p->velocity_max_rad_s);
        crc_f32(&crc, p->effort_max_nm);
        crc_f32(&crc, p->safe_damping_nm_s_rad);
        crc_f32(&crc, p->safe_damping_stop_velocity_rad_s);
        crc_f32(&crc, p->effort_to_backend);
        crc_u16(&crc, p->watchdog_ms);
        crc_u16(&crc, p->safe_damping_duration_ms);
        crc_u16(&crc, p->capabilities);
    }
    return esd_link_v1_crc32_end(crc);
}

float esd_port_joint_effort(const esd_port_config_t *port,
                            const esd_port_feedback_t *feedback,
                            const esd_port_command_t *command)
{
    if ((port == NULL) || (feedback == NULL) || (command == NULL)) {
        return 0.0f;
    }
    const float effort = command->kp_nm_rad *
                             (command->q_des_rad - feedback->position_rad) +
                         command->kd_nm_s_rad *
                             (command->dq_des_rad_s - feedback->velocity_rad_s) +
                         command->tau_ff_nm;
    return clampf(effort, -port->effort_max_nm, port->effort_max_nm);
}

bool esd_port_apply_effort(const esd_port_config_t *port, float joint_effort)
{
    if ((port == NULL) || !isfinite(joint_effort)) {
        return false;
    }
    const float motor_effort = joint_effort * (float)port->direction /
                               (port->gear_ratio * port->motor_efficiency);
    return port->ops->set_motor_effort(
        port->user, motor_effort * port->effort_to_backend);
}

bool esd_port_apply_command(const esd_port_config_t *port,
                            const esd_port_feedback_t *feedback,
                            const esd_port_command_t *command,
                            float software_zero_rad)
{
    if ((port == NULL) || (feedback == NULL) || (command == NULL) ||
        !isfinite(feedback->position_rad) ||
        !isfinite(feedback->velocity_rad_s) ||
        !isfinite(command->q_des_rad) ||
        !isfinite(command->dq_des_rad_s) ||
        !isfinite(command->kp_nm_rad) ||
        !isfinite(command->kd_nm_s_rad) ||
        !isfinite(command->tau_ff_nm) ||
        !isfinite(software_zero_rad) ||
        (command->kp_nm_rad < 0.0f) ||
        (command->kd_nm_s_rad < 0.0f)) {
        return false;
    }

    if (port->ops->set_motor_command == NULL) {
        const float effort = esd_port_joint_effort(port, feedback, command);
        return isfinite(effort) && esd_port_apply_effort(port, effort);
    }

    const float p_effort = command->kp_nm_rad *
                           (command->q_des_rad - feedback->position_rad);
    const float d_effort = command->kd_nm_s_rad *
                           (command->dq_des_rad_s - feedback->velocity_rad_s);
    const float requested = fabsf(p_effort) + fabsf(d_effort) +
                            fabsf(command->tau_ff_nm);
    if (!isfinite(requested)) {
        return false;
    }
    const float scale = (requested > port->effort_max_nm)
                            ? (port->effort_max_nm / requested)
                            : 1.0f;
    const float gain_to_motor = port->effort_to_backend /
                                (port->gear_ratio * port->gear_ratio *
                                 port->motor_efficiency);
    const float effort_to_motor = (float)port->direction *
                                  port->effort_to_backend /
                                  (port->gear_ratio *
                                   port->motor_efficiency);
    const esd_port_command_t motor_command = {
        .q_des_rad = (float)port->direction * port->gear_ratio *
                     (command->q_des_rad + software_zero_rad),
        .dq_des_rad_s = (float)port->direction * port->gear_ratio *
                        command->dq_des_rad_s,
        .kp_nm_rad = command->kp_nm_rad * scale * gain_to_motor,
        .kd_nm_s_rad = command->kd_nm_s_rad * scale * gain_to_motor,
        .tau_ff_nm = command->tau_ff_nm * scale * effort_to_motor,
    };
    return port->ops->set_motor_command(port->user, &motor_command);
}

void esd_port_disable_all(const esd_port_registry_t *registry)
{
    if (registry == NULL) {
        return;
    }
    for (size_t i = 0u; i < registry->count; ++i) {
        const esd_port_config_t *port = &registry->ports[i];
        port->ops->emergency_disable(port->user);
    }
}
