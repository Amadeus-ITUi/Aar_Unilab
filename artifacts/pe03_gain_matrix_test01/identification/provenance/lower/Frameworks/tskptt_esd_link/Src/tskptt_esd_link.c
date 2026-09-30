#include "tskptt_esd_link.h"

#include "cmsis_os2.h"

#include <math.h>
#include <string.h>

#define NOTIFY_RX              (1u << 0)
#define NOTIFY_STATE           (1u << 1)
#define CFG_QUEUE_DEPTH        2u
#define CFG_RESPONSE_DEPTH     2u
#define CONFIG_RESPONSE_LEN    ESD_LINK_V1_TRANSACTION_RESPONSE_BODY_LEN
#define DEVICE_PERIOD_MS       100u
#define ENABLE_VERIFY_MS       100u
#define STATE_SOURCE_HISTORY_COUNT TSKPTT_ESD_LINK_STATE_HISTORY_COUNT
#define COMMAND_STATUS_HAS_APPLIED       (1u << 0)
#define COMMAND_STATUS_SAFE_DAMPING      (1u << 2)
#define PORT_CONFIG_TIMEOUT_MS            5000u

typedef struct {
    uint8_t service;
    uint8_t message;
    uint16_t length;
    uint32_t sequence;
    uint8_t payload[ESD_LINK_V1_MAX_PAYLOAD];
} config_work_t;

typedef struct {
    uint8_t service;
    uint8_t message;
    uint8_t flags;
    uint8_t reserved;
    uint16_t status;
    uint16_t body_len;
    uint32_t sequence;
    uint8_t body[CONFIG_RESPONSE_LEN];
} config_response_t;

typedef struct {
    uint8_t used;
    uint8_t pending;
    uint8_t service;
    uint8_t message;
    uint32_t session;
    uint32_t transaction;
    uint32_t request_crc;
    uint16_t status;
    uint8_t body_len;
    uint8_t body[CONFIG_RESPONSE_LEN];
} transaction_entry_t;

typedef struct {
    uint32_t sequence;
    uint32_t received_ms;
    uint32_t source_state_sequence;
    uint8_t count;
    uint8_t valid;
    esd_port_command_t command[ESD_PORT_ACTIVE_MAX];
} command_snapshot_t;

typedef struct {
    uint32_t sample_sequence;
    uint32_t sample_time_us;
    uint8_t count;
    uint8_t imu_valid;
    float accel[3];
    float gyro[3];
    float quat[4];
    esd_port_feedback_t feedback[ESD_PORT_ACTIVE_MAX];
} feedback_snapshot_t;

typedef struct {
    feedback_snapshot_t feedback;
    command_snapshot_t command;
    uint32_t session_id;
    uint32_t fault_flags;
    uint32_t active_port_mask;
    uint32_t offline_port_mask;
    uint32_t valid_command_count;
    uint32_t rejected_command_count;
    uint16_t last_reject_code;
    uint16_t command_age_ms;
    uint8_t control_state;
    uint8_t output_gate;
} control_cycle_snapshot_t;

typedef struct {
    uint32_t config_id;
    uint32_t table_crc;
    uint32_t received_mask;
    uint32_t last_activity_ms;
    uint8_t count;
    uint8_t active;
    uint8_t raw[ESD_PORT_ACTIVE_MAX][ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN];
} port_config_staging_t;

static ESD_LINK_ALIGN32 esd_link_t s_link;
static tskptt_esd_link_config_t s_cfg;
static uint16_t s_control_rate_hz;
static uint32_t s_control_period_ms;
static esd_port_registry_t s_ports;
static osMessageQueueId_t s_cfg_queue;
static osMessageQueueId_t s_response_queue;
static TaskHandle_t s_link_task;
static TaskHandle_t s_ctrl_task;
static TaskHandle_t s_config_task;
static transaction_entry_t s_transactions[TSKPTT_ESD_LINK_TRANSACTION_COUNT];
static command_snapshot_t s_command;
static control_cycle_snapshot_t s_cycle;
static float s_zero[ESD_PORT_ID_COUNT];
static volatile uint32_t s_session;
static volatile uint32_t s_disconnect_generation;
static volatile uint32_t s_control_deadline_miss;
static volatile uint32_t s_last_command_ms;
static volatile uint32_t s_valid_commands;
static volatile uint32_t s_rejected_commands;
static volatile uint16_t s_last_reject;
static volatile uint8_t s_control_state;
static volatile uint8_t s_output_gate;
static volatile uint8_t s_control_fault_latched;
static uint32_t s_seen_deadline_miss;
static uint32_t s_event_sequence;
static uint32_t s_state_sequence;
static uint32_t s_safe_damping_started_ms;
static volatile uint8_t s_safe_damping_terminal_state;
static uint32_t s_published_state_sequence[STATE_SOURCE_HISTORY_COUNT];
static uint8_t s_published_state_count;
static uint8_t s_published_state_next;
static uint32_t s_last_rx_overflow;
static uint32_t s_config_fingerprint;
static uint32_t s_hskp_period_ms;
static uint32_t s_hskp_last_ms;
static uint8_t s_connected;
static uint8_t s_configured;
static uint8_t s_ready;
static uint32_t s_link_disconnect_generation;
static uint32_t s_ctrl_disconnect_generation;
static uint32_t s_last_status_fault_flags;
static uint32_t s_last_status_offline_mask;
static uint16_t s_last_status_reject;
static uint8_t s_last_status_control_state;
static uint8_t s_status_signature_valid;
static port_config_staging_t s_port_staging;

static void copy_bytes(uint8_t *dst, const uint8_t *src, size_t length)
{
    volatile uint8_t *d = dst;
    const volatile uint8_t *s = src;
    while (length-- != 0u) *d++ = *s++;
}

static uint32_t now_ms(void)
{
    return s_cfg.now_ms(s_cfg.clock_user);
}

static uint32_t cycles(void)
{
    return (s_cfg.cycles != NULL) ? s_cfg.cycles(s_cfg.clock_user) : 0u;
}

static size_t bounded_strlen(const char *text, size_t limit)
{
    size_t length = 0u;
    while ((length < limit) && (text[length] != '\0')) {
        ++length;
    }
    return length;
}

static uint32_t link_now(void *user) { (void)user; return now_ms(); }
static uint32_t link_cycles(void *user) { (void)user; return cycles(); }
static void link_lock(void *user) { (void)user; taskENTER_CRITICAL(); }
static void link_unlock(void *user) { (void)user; taskEXIT_CRITICAL(); }
static esd_link_tx_result_t link_submit(void *user, const uint8_t *data, size_t len)
{ (void)user; return s_cfg.transport.tx_submit(s_cfg.transport.user, data, len); }

static uint32_t active_mask(void)
{
    uint32_t mask = 0u;
    for (size_t i = 0u; i < s_ports.count; ++i) mask |= 1u << s_ports.ports[i].port_id;
    return mask;
}

static uint32_t refresh_fingerprint(void)
{
    s_config_fingerprint =
        esd_port_compute_fingerprint_with_offsets(&s_ports, s_zero);
    return s_config_fingerprint;
}

static void clear_published_state_history(void)
{
    taskENTER_CRITICAL();
    memset(s_published_state_sequence, 0, sizeof(s_published_state_sequence));
    s_published_state_count = 0u;
    s_published_state_next = 0u;
    taskEXIT_CRITICAL();
}

static void remember_published_state(uint32_t sequence)
{
    taskENTER_CRITICAL();
    s_published_state_sequence[s_published_state_next] = sequence;
    s_published_state_next =
        (uint8_t)((s_published_state_next + 1u) % STATE_SOURCE_HISTORY_COUNT);
    if (s_published_state_count < STATE_SOURCE_HISTORY_COUNT) {
        s_published_state_count++;
    }
    taskEXIT_CRITICAL();
}

static bool is_recent_published_state(uint32_t sequence)
{
    bool found = false;
    taskENTER_CRITICAL();
    for (uint8_t i = 0u; i < s_published_state_count; ++i) {
        if (s_published_state_sequence[i] == sequence) {
            found = true;
            break;
        }
    }
    taskEXIT_CRITICAL();
    return found;
}

static void invalidate_session(tskptt_esd_control_state_t next)
{
    taskENTER_CRITICAL();
    s_output_gate = 0u;
    s_session = 0u;
    s_command.valid = 0u;
    s_control_state = (uint8_t)next;
    s_safe_damping_started_ms = 0u;
    s_safe_damping_terminal_state = TSKPTT_ESD_CONTROL_SAFE_DAMPING;
    memset(s_transactions, 0, sizeof(s_transactions));
    taskEXIT_CRITICAL();
    clear_published_state_history();
}

static void emergency_disable(tskptt_esd_control_state_t next)
{
    taskENTER_CRITICAL();
    s_output_gate = 0u;
    s_command.valid = 0u;
    s_control_state = (uint8_t)next;
    s_safe_damping_started_ms = 0u;
    s_safe_damping_terminal_state = TSKPTT_ESD_CONTROL_SAFE_DAMPING;
    taskEXIT_CRITICAL();
    esd_port_disable_all(&s_ports);
}

static void begin_safe_damping(uint32_t now,
                               tskptt_esd_control_state_t terminal_state)
{
    taskENTER_CRITICAL();
    s_command.valid = 0u;
    if (s_control_state != TSKPTT_ESD_CONTROL_SAFE_DAMPING) {
        s_safe_damping_started_ms = now;
    }
    s_control_state = TSKPTT_ESD_CONTROL_SAFE_DAMPING;
    s_safe_damping_terminal_state = (uint8_t)terminal_state;
    taskEXIT_CRITICAL();
}

static uint16_t safe_damping_duration_ms(void)
{
    uint16_t duration = 0u;
    for (size_t i = 0u; i < s_ports.count; ++i) {
        if (s_ports.ports[i].safe_damping_duration_ms > duration) {
            duration = s_ports.ports[i].safe_damping_duration_ms;
        }
    }
    return duration;
}

static bool apply_safe_damping(const feedback_snapshot_t *feedback,
                               bool *all_stopped)
{
    if ((feedback == NULL) || (all_stopped == NULL) ||
        (feedback->count != s_ports.count)) {
        return false;
    }
    bool stopped = true;
    for (size_t i = 0u; i < s_ports.count; ++i) {
        const esd_port_config_t *port = &s_ports.ports[i];
        const esd_port_feedback_t *sample = &feedback->feedback[i];
        if (!sample->valid || !sample->online || sample->fault ||
            !isfinite(sample->velocity_rad_s)) {
            return false;
        }
        if (fabsf(sample->velocity_rad_s) >
            port->safe_damping_stop_velocity_rad_s) {
            stopped = false;
        }
        float effort = -port->safe_damping_nm_s_rad *
                       sample->velocity_rad_s;
        if (effort > port->effort_max_nm) effort = port->effort_max_nm;
        if (effort < -port->effort_max_nm) effort = -port->effort_max_nm;
        if (!esd_port_apply_effort(port, effort)) return false;
    }
    *all_stopped = stopped;
    return true;
}

static bool run_safe_damping(uint32_t now,
                             const feedback_snapshot_t *feedback)
{
    bool all_stopped = false;
    if (!apply_safe_damping(feedback, &all_stopped)) return false;
    if (all_stopped ||
        ((uint32_t)(now - s_safe_damping_started_ms) >=
         safe_damping_duration_ms())) {
        const tskptt_esd_control_state_t terminal_state =
            (tskptt_esd_control_state_t)s_safe_damping_terminal_state;
        emergency_disable(terminal_state);
    }
    return true;
}

static void disconnect_safe_path(uint32_t now)
{
    bool use_damping;
    taskENTER_CRITICAL();
    use_damping = s_output_gate != 0u;
    s_session = 0u;
    s_command.valid = 0u;
    memset(s_transactions, 0, sizeof(s_transactions));
    if (use_damping) {
        if (s_control_state != TSKPTT_ESD_CONTROL_SAFE_DAMPING) {
            s_safe_damping_started_ms = now;
        }
        s_control_state = TSKPTT_ESD_CONTROL_SAFE_DAMPING;
        s_safe_damping_terminal_state = TSKPTT_ESD_CONTROL_SAFE_DAMPING;
    } else {
        s_control_state = TSKPTT_ESD_CONTROL_NO_SESSION;
    }
    taskEXIT_CRITICAL();
    clear_published_state_history();
    if (!use_damping) esd_port_disable_all(&s_ports);
}

static transaction_entry_t *transaction_find(uint8_t service, uint8_t message,
                                              uint32_t session, uint32_t tx)
{
    for (size_t i=0u;i<TSKPTT_ESD_LINK_TRANSACTION_COUNT;++i) {
        transaction_entry_t *e=&s_transactions[i];
        if (e->used && e->service==service && e->message==message &&
            e->session==session && e->transaction==tx) return e;
    }
    return NULL;
}

static transaction_entry_t *transaction_allocate(void)
{
    static uint8_t cursor;
    for (size_t i=0u;i<TSKPTT_ESD_LINK_TRANSACTION_COUNT;++i)
        if (!s_transactions[i].used) return &s_transactions[i];
    for (size_t i=0u;i<TSKPTT_ESD_LINK_TRANSACTION_COUNT;++i) {
        transaction_entry_t *e=&s_transactions[cursor++ % TSKPTT_ESD_LINK_TRANSACTION_COUNT];
        if (!e->pending) { memset(e,0,sizeof(*e)); return e; }
    }
    return NULL;
}

static void response_body(uint8_t out[CONFIG_RESPONSE_LEN], uint32_t requested_session,
                          uint32_t transaction, uint8_t failed, uint16_t detail,
                          uint32_t requested, uint32_t affected, uint32_t verified,
                          uint32_t rollback)
{
    esd_link_v1_le32_write(&out[0], requested_session);
    esd_link_v1_le32_write(&out[4], s_session);
    esd_link_v1_le32_write(&out[8], transaction);
    out[12]=failed;
    esd_link_v1_le16_write(&out[13],detail);
    esd_link_v1_le32_write(&out[15],s_config_fingerprint);
    esd_link_v1_le32_write(&out[19],requested);
    esd_link_v1_le32_write(&out[23],affected);
    esd_link_v1_le32_write(&out[27],verified);
    esd_link_v1_le32_write(&out[31],rollback);
}

static uint16_t check_session(const esd_link_v1_frame_view_t *request)
{
    if (request->payload_len < 4u) return ESD_LINK_STATUS_INVALID_LENGTH;
    return (esd_link_v1_le32_read(request->payload) == s_session && s_session != 0u) ?
        ESD_LINK_STATUS_OK : ESD_LINK_STATUS_SESSION_MISMATCH;
}

static uint16_t handle_open(void *user, const esd_link_v1_frame_view_t *request,
                            uint8_t *response, size_t cap, size_t *len, uint8_t *flags)
{
    (void)user; *flags=0u; *len=0u;
    if ((request->payload_len != 17u) || (cap < 32u)) return ESD_LINK_STATUS_INVALID_LENGTH;
    const uint32_t nonce=esd_link_v1_le32_read(&request->payload[0]);
    const uint8_t version=request->payload[4];
    const uint16_t layout=esd_link_v1_le16_read(&request->payload[5]);
    const uint16_t schema=esd_link_v1_le16_read(&request->payload[7]);
    const uint32_t expected=esd_link_v1_le32_read(&request->payload[9]);
    uint16_t status=ESD_LINK_STATUS_OK;
    if ((nonce==0u)||(version!=1u)) status=ESD_LINK_STATUS_UNSUPPORTED_VERSION;
    else if (!s_cfg.configuration_healthy) status=ESD_LINK_STATUS_CONFIG_MISMATCH;
    else if (layout!=s_cfg.layout_id) status=ESD_LINK_STATUS_LAYOUT_MISMATCH;
    else if (schema!=s_cfg.schema_id) status=ESD_LINK_STATUS_SCHEMA_MISMATCH;
    else if ((expected!=0u)&&(expected!=s_config_fingerprint)) status=ESD_LINK_STATUS_CONFIG_MISMATCH;
    emergency_disable((s_cfg.configuration_healthy&&s_cfg.boot_storage_healthy)?
        TSKPTT_ESD_CONTROL_NO_SESSION:TSKPTT_ESD_CONTROL_FAULT_LATCHED);
    uint32_t session=0u;
    if (status==ESD_LINK_STATUS_OK) {
        uint8_t seed[12];
        esd_link_v1_le32_write(&seed[0],s_cfg.lower_boot_id);
        esd_link_v1_le32_write(&seed[4],nonce);
        esd_link_v1_le32_write(&seed[8],s_config_fingerprint);
        session=esd_link_v1_crc32(seed,sizeof(seed));
        if (session==0u) session=1u;
        clear_published_state_history();
        taskENTER_CRITICAL(); memset(s_transactions,0,sizeof(s_transactions)); s_control_fault_latched=0u; s_seen_deadline_miss=s_control_deadline_miss; s_session=session; s_control_state=s_cfg.boot_storage_healthy?TSKPTT_ESD_CONTROL_DISABLED:TSKPTT_ESD_CONTROL_FAULT_LATCHED; taskEXIT_CRITICAL();
    }
    esd_link_v1_le32_write(&response[0],nonce);
    esd_link_v1_le32_write(&response[4],s_cfg.lower_boot_id);
    esd_link_v1_le32_write(&response[8],session);
    response[12]=1u;
    esd_link_v1_le16_write(&response[13],s_cfg.layout_id);
    esd_link_v1_le32_write(&response[15],active_mask());
    esd_link_v1_le16_write(&response[19],s_cfg.schema_id);
    esd_link_v1_le32_write(&response[21],s_config_fingerprint);
    esd_link_v1_le16_write(&response[25],s_control_rate_hz);
    esd_link_v1_le32_write(&response[27],s_cfg.device_capabilities);
    response[31]=s_control_state;
    *len=32u;
    return status;
}

static bool get_f32(const uint8_t *p, float *out)
{
    uint32_t bits=esd_link_v1_le32_read(p);
    memcpy(out,&bits,sizeof(bits));
    return isfinite(*out);
}

static uint16_t handle_command(void *user, const esd_link_v1_frame_view_t *request,
                               uint8_t *response, size_t cap, size_t *len, uint8_t *flags)
{
    (void)user;(void)response;(void)cap;*len=0u;*flags=0u;
    uint16_t reject=ESD_LINK_STATUS_OK;
    if (request->payload_len < 15u) reject=ESD_LINK_STATUS_INVALID_LENGTH;
    else if (check_session(request)!=ESD_LINK_STATUS_OK) reject=ESD_LINK_STATUS_SESSION_MISMATCH;
    else if ((s_control_state!=TSKPTT_ESD_CONTROL_ENABLED_WAIT_COMMAND)&&
             (s_control_state!=TSKPTT_ESD_CONTROL_ACTIVE)) reject=ESD_LINK_STATUS_INVALID_STATE;
    const uint8_t count=(request->payload_len>=15u)?request->payload[14]:0u;
    if ((reject==0u)&&((count==0u)||(count!=s_ports.count)||
        (request->payload_len != (size_t)(15u+21u*count)))) reject=ESD_LINK_STATUS_INVALID_LENGTH;
    if ((reject==0u)&&(esd_link_v1_le16_read(&request->payload[4])!=s_cfg.layout_id)) reject=ESD_LINK_STATUS_LAYOUT_MISMATCH;
    if ((reject==0u)&&(esd_link_v1_le32_read(&request->payload[10])!=s_config_fingerprint)) reject=ESD_LINK_STATUS_CONFIG_MISMATCH;
    const uint32_t source=(request->payload_len>=10u)?esd_link_v1_le32_read(&request->payload[6]):0u;
    if ((reject==0u)&&!is_recent_published_state(source)) reject=ESD_LINK_STATUS_INVALID_VALUE;
    command_snapshot_t next; memset(&next,0,sizeof(next));
    next.count=count; next.sequence=request->header.sequence; next.received_ms=now_ms(); next.source_state_sequence=source;
    uint32_t seen=0u;
    for (size_t i=0u;(reject==0u)&&(i<count);++i) {
        const uint8_t *p=&request->payload[15u+21u*i];
        const uint8_t id=p[0];
        const esd_port_config_t *port=esd_port_find(&s_ports,id);
        if (port==NULL) { reject=ESD_LINK_STATUS_INVALID_PORT_ID; break; }
        if ((seen&(1u<<id))!=0u) { reject=ESD_LINK_STATUS_DUPLICATE_PORT_ID; break; }
        seen|=1u<<id;
        size_t port_index=0u;
        while(port_index<s_ports.count && s_ports.ports[port_index].port_id!=id)port_index++;
        if(port_index==s_ports.count){reject=ESD_LINK_STATUS_INVALID_PORT_ID;break;}
        esd_port_command_t *c=&next.command[port_index];
        if (!get_f32(&p[1],&c->q_des_rad)||!get_f32(&p[5],&c->dq_des_rad_s)||
            !get_f32(&p[9],&c->kp_nm_rad)||!get_f32(&p[13],&c->kd_nm_s_rad)||
            !get_f32(&p[17],&c->tau_ff_nm)||c->kp_nm_rad<0.0f||c->kd_nm_s_rad<0.0f||
            c->q_des_rad<port->position_min_rad||c->q_des_rad>port->position_max_rad||
            fabsf(c->dq_des_rad_s)>port->velocity_max_rad_s||
            fabsf(c->tau_ff_nm)>port->effort_max_nm) reject=ESD_LINK_STATUS_INVALID_VALUE;
    }
    if ((reject==0u)&&(seen!=active_mask())) reject=ESD_LINK_STATUS_INVALID_PORT_ID;
    if ((reject==0u)&&s_command.valid) {
        const uint32_t delta=request->header.sequence-s_command.sequence;
        if ((delta==0u)||(delta>0x7FFFFFFFu)) reject=ESD_LINK_STATUS_CONFLICT;
    }
    if (reject==0u) {
        next.valid=1u;
        taskENTER_CRITICAL(); s_command=next; s_last_command_ms=next.received_ms; s_valid_commands++; taskEXIT_CRITICAL();
    } else { s_last_reject=reject; s_rejected_commands++; }
    return reject;
}

static uint16_t enqueue_config(void *user, const esd_link_v1_frame_view_t *request,
                               uint8_t *response, size_t cap, size_t *len, uint8_t *flags)
{
    (void)user;*flags=0u;*len=0u;
    if ((request->payload_len<8u)||(request->payload_len>ESD_LINK_V1_MAX_PAYLOAD)) return ESD_LINK_STATUS_INVALID_LENGTH;
    const uint32_t session=esd_link_v1_le32_read(request->payload);
    const uint32_t tx=esd_link_v1_le32_read(&request->payload[4]);
    const uint32_t request_crc=esd_link_v1_crc32(request->payload,request->payload_len);
    if ((session!=s_session)||(session==0u)) {
        if (cap>=CONFIG_RESPONSE_LEN) { response_body(response,session,tx,0xFFu,0u,0u,0u,0u,0u); *len=CONFIG_RESPONSE_LEN; }
        return ESD_LINK_STATUS_SESSION_MISMATCH;
    }
    transaction_entry_t *old=transaction_find(request->header.service,request->header.message,session,tx);
    if (old!=NULL) {
        if(old->request_crc!=request_crc)return ESD_LINK_STATUS_CONFLICT;
        if (old->pending) return ESD_LINK_STATUS_BUSY;
        if (cap<old->body_len) return ESD_LINK_STATUS_INTERNAL_ERROR;
        copy_bytes(response,old->body,old->body_len);*len=old->body_len;return old->status;
    }
    config_work_t work={.service=request->header.service,.message=request->header.message,
                        .length=(uint16_t)request->payload_len,.sequence=request->header.sequence};
    copy_bytes(work.payload,request->payload,request->payload_len);
    transaction_entry_t *entry=transaction_allocate();
    if(entry==NULL)return ESD_LINK_STATUS_BUSY;
    memset(entry,0,sizeof(*entry)); entry->used=1u;entry->pending=1u;entry->service=work.service;
    entry->message=work.message;entry->session=session;entry->transaction=tx;entry->request_crc=request_crc;
    if (osMessageQueuePut(s_cfg_queue,&work,0u,0u)!=osOK) { entry->used=0u; return ESD_LINK_STATUS_BUSY; }
    return ESD_LINK_STATUS_DEFERRED_INTERNAL;
}

static uint16_t handle_enable(void *u,const esd_link_v1_frame_view_t *r,uint8_t *o,size_t c,size_t *l,uint8_t *f)
{
    if ((r->payload_len>=10u)&&(r->payload[8]==0u)) {
        const uint32_t session=esd_link_v1_le32_read(r->payload);
        const uint32_t tx=esd_link_v1_le32_read(&r->payload[4]);
        if ((session!=s_session)||(session==0u)) return enqueue_config(u,r,o,c,l,f);
        const uint8_t count=r->payload[9];uint32_t requested=0u;
        if((count==0u)||(count>ESD_PORT_ACTIVE_MAX)||(r->payload_len!=(size_t)(10u+count)))return ESD_LINK_STATUS_INVALID_LENGTH;
        if((count==1u)&&(r->payload[10]==0xFFu))requested=active_mask();
        else {for(size_t i=0u;i<count;++i){uint8_t id=r->payload[10u+i];if(esd_port_find(&s_ports,id)==NULL)return ESD_LINK_STATUS_INVALID_PORT_ID;if(requested&(1u<<id))return ESD_LINK_STATUS_DUPLICATE_PORT_ID;requested|=1u<<id;}if(requested!=active_mask())return ESD_LINK_STATUS_INVALID_STATE;}
        const uint32_t request_crc=esd_link_v1_crc32(r->payload,r->payload_len);
        transaction_entry_t *old=transaction_find(r->header.service,r->header.message,session,tx);
        if(old!=NULL){if(old->request_crc!=request_crc)return ESD_LINK_STATUS_CONFLICT;if(old->pending)return ESD_LINK_STATUS_BUSY;if(c<old->body_len)return ESD_LINK_STATUS_INTERNAL_ERROR;copy_bytes(o,old->body,old->body_len);*l=old->body_len;*f=0u;return old->status;}
        bool use_damping;
        taskENTER_CRITICAL();use_damping=s_output_gate!=0u;taskEXIT_CRITICAL();
        if(use_damping)begin_safe_damping(now_ms(),TSKPTT_ESD_CONTROL_DISABLED);
        else emergency_disable(TSKPTT_ESD_CONTROL_DISABLED);
        if (c<CONFIG_RESPONSE_LEN) return ESD_LINK_STATUS_INTERNAL_ERROR;
        uint32_t disabled_verified=0u;if(!use_damping)for(size_t i=0u;i<s_ports.count;++i)if(s_ports.ports[i].ops->verify_disabled(s_ports.ports[i].user))disabled_verified|=1u<<s_ports.ports[i].port_id;
        response_body(o,session,tx,0xFFu,0u,requested,active_mask(),disabled_verified,0u);
        transaction_entry_t *entry=transaction_allocate();if(entry==NULL)return ESD_LINK_STATUS_BUSY;memset(entry,0,sizeof(*entry));entry->used=1u;entry->service=r->header.service;entry->message=r->header.message;entry->session=session;entry->transaction=tx;entry->request_crc=request_crc;entry->status=ESD_LINK_STATUS_OK;entry->body_len=CONFIG_RESPONSE_LEN;copy_bytes(entry->body,o,CONFIG_RESPONSE_LEN);
        *l=CONFIG_RESPONSE_LEN;*f=0u;return ESD_LINK_STATUS_OK;
    }
    return enqueue_config(u,r,o,c,l,f);
}

static uint16_t handle_hskp(void *user,const esd_link_v1_frame_view_t *r,uint8_t *o,size_t cap,size_t *len,uint8_t *flags)
{
    (void)user;*flags=0u;*len=0u;
    if ((r->payload_len!=8u)||(cap<8u)) return ESD_LINK_STATUS_INVALID_LENGTH;
    if (check_session(r)!=ESD_LINK_STATUS_OK) return ESD_LINK_STATUS_SESSION_MISMATCH;
    const uint32_t period=esd_link_v1_le32_read(&r->payload[4]);
    if ((period!=0u)&&((period<20u)||(period>60000u))) return ESD_LINK_STATUS_INVALID_VALUE;
    s_hskp_period_ms=period;s_hskp_last_ms=0u;
    if(s_cfg.hskp_config)s_cfg.hskp_config(s_cfg.project_user,period!=0u,period*1000u);
    copy_bytes(o,r->payload,8u);*len=8u;return ESD_LINK_STATUS_OK;
}

static esd_link_result_t start_link(void)
{
    if (!s_cfg.transport.init(s_cfg.transport.user)) return ESD_LINK_ERR_TRANSPORT;
    esd_link_config_t lc={.now_ms=link_now,.cycles=(s_cfg.cycles?link_cycles:NULL),
        .tx_submit=link_submit,.lock=link_lock,.unlock=link_unlock,
        .tx_storage=s_cfg.tx_storage,.tx_storage_size=s_cfg.tx_storage_size,.user=NULL};
    esd_link_result_t result=esd_link_init(&s_link,&lc);
    if (result!=ESD_LINK_OK) return result;
#define REG(s,m,k,q,h) do { result=esd_link_register_handler(&s_link,s,m,ESD_LINK_KIND_MASK(k),q,h,NULL); if(result!=ESD_LINK_OK)return result; } while(0)
    REG(ESD_LINK_SERVICE_SESSION,ESD_LINK_SESSION_OPEN,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,handle_open);
    REG(ESD_LINK_SERVICE_CONTROL,ESD_LINK_CONTROL_ACTUATOR_COMMAND,ESD_LINK_V1_KIND_EVENT,ESD_LINK_QOS_REALTIME,handle_command);
    REG(ESD_LINK_SERVICE_CONTROL,ESD_LINK_CONTROL_SET_ENABLE,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,handle_enable);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_SET_ZERO,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_PORT_CONFIG_BEGIN,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_PORT_CONFIG_CHUNK,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_PORT_CONFIG_COMMIT,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_PORT_CONFIG_ABORT,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_CONFIG,ESD_LINK_CONFIG_PORT_CONFIG_UPDATE,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,enqueue_config);
    REG(ESD_LINK_SERVICE_HSKP,ESD_LINK_HSKP_SUBSCRIBE,ESD_LINK_V1_KIND_REQUEST,ESD_LINK_QOS_RELIABLE,handle_hskp);
#undef REG
    if ((esd_link_register_tx_topic(&s_link,ESD_LINK_SERVICE_STATE,ESD_LINK_STATE_ROBOT_STATE,ESD_LINK_QOS_REALTIME)!=ESD_LINK_OK)||
        (esd_link_register_tx_topic(&s_link,ESD_LINK_SERVICE_STATE,ESD_LINK_STATE_DEVICE_STATUS,ESD_LINK_QOS_DIAGNOSTIC)!=ESD_LINK_OK)||
        (esd_link_register_tx_topic(&s_link,ESD_LINK_SERVICE_HSKP,ESD_LINK_HSKP_REPORT,ESD_LINK_QOS_DIAGNOSTIC)!=ESD_LINK_OK)||
        (esd_link_register_tx_topic(&s_link,ESD_LINK_SERVICE_LOG,ESD_LINK_LOG_RECORD,ESD_LINK_QOS_BEST_EFFORT)!=ESD_LINK_OK)) return ESD_LINK_ERR_FULL;
    if (!s_cfg.transport.start_rx(s_cfg.transport.user,s_link_task,NOTIFY_RX)) return ESD_LINK_ERR_TRANSPORT;
    s_ready=1u;return ESD_LINK_OK;
}

static void put_f32(uint8_t *p,float value)
{ uint32_t bits;memcpy(&bits,&value,sizeof(bits));esd_link_v1_le32_write(p,bits); }

static void publish_state(uint32_t now)
{
    control_cycle_snapshot_t cycle;
    taskENTER_CRITICAL();cycle=s_cycle;taskEXIT_CRITICAL();
    if (cycle.session_id==0u) return;
    const feedback_snapshot_t *snap=&cycle.feedback;
    const command_snapshot_t *command=&cycle.command;
    uint8_t p[ESD_LINK_V1_MAX_PAYLOAD];size_t n=0u;
    esd_link_v1_le32_write(&p[n],cycle.session_id);n+=4;esd_link_v1_le16_write(&p[n],s_cfg.schema_id);n+=2;
    esd_link_v1_le32_write(&p[n],snap->sample_time_us);n+=4;esd_link_v1_le32_write(&p[n],snap->sample_sequence);n+=4;
    esd_link_v1_le32_write(&p[n],command->valid?command->sequence:0u);n+=4;
    uint8_t command_status=command->valid?COMMAND_STATUS_HAS_APPLIED:0u;
    if(cycle.control_state==TSKPTT_ESD_CONTROL_SAFE_DAMPING&&cycle.output_gate)command_status|=COMMAND_STATUS_SAFE_DAMPING;
    p[n++]=command_status;p[n++]=cycle.control_state;p[n++]=(s_cfg.schema_id==2u&&snap->count<=11u&&command->valid)?4u:3u;
    p[n++]=1u;esd_link_v1_le16_write(&p[n],ESD_LINK_V1_IMU_BLOCK_LEN);n+=2;p[n++]=snap->imu_valid;
    for(size_t i=0u;i<3u;++i){put_f32(&p[n],snap->gyro[i]);n+=4;}for(size_t i=0u;i<4u;++i){put_f32(&p[n],snap->quat[i]);n+=4;}
    for(size_t i=0u;i<3u;++i){put_f32(&p[n],snap->accel[i]);n+=4;}
    p[n++]=2u;esd_link_v1_le16_write(&p[n],(uint16_t)(1u+10u*snap->count));n+=2;p[n++]=snap->count;
    for(size_t i=0u;i<snap->count;++i){p[n++]=s_ports.ports[i].port_id;p[n++]=snap->feedback[i].valid?3u:0u;put_f32(&p[n],snap->feedback[i].position_rad);n+=4;put_f32(&p[n],snap->feedback[i].velocity_rad_s);n+=4;}
    p[n++]=3u;esd_link_v1_le16_write(&p[n],(uint16_t)(1u+5u*snap->count));n+=2;p[n++]=snap->count;
    for(size_t i=0u;i<snap->count;++i){p[n++]=s_ports.ports[i].port_id;put_f32(&p[n],snap->feedback[i].effort_nm);n+=4;}
    if ((s_cfg.schema_id==2u)&&(snap->count<=11u)&&command->valid) {
        p[n++]=4u;esd_link_v1_le16_write(&p[n],(uint16_t)(1u+21u*snap->count));n+=2;p[n++]=snap->count;
        for(size_t i=0u;i<snap->count;++i){const esd_port_command_t *c=&command->command[i];p[n++]=s_ports.ports[i].port_id;
            put_f32(&p[n],c->q_des_rad);n+=4;put_f32(&p[n],c->dq_des_rad_s);n+=4;put_f32(&p[n],c->kp_nm_rad);n+=4;put_f32(&p[n],c->kd_nm_s_rad);n+=4;put_f32(&p[n],c->tau_ff_nm);n+=4;}
    }
    if (esd_link_publish(&s_link,ESD_LINK_SERVICE_STATE,ESD_LINK_STATE_ROBOT_STATE,
                         0u,++s_event_sequence,p,n)==ESD_LINK_OK) {
        remember_published_state(snap->sample_sequence);
    }
    (void)now;
}

static control_cycle_snapshot_t cycle_snapshot(void)
{
    control_cycle_snapshot_t cycle;
    taskENTER_CRITICAL();cycle=s_cycle;taskEXIT_CRITICAL();
    return cycle;
}

static bool status_changed(const control_cycle_snapshot_t *cycle)
{
    if(s_status_signature_valid&&s_last_status_fault_flags==cycle->fault_flags&&
       s_last_status_offline_mask==cycle->offline_port_mask&&
       s_last_status_reject==cycle->last_reject_code&&
       s_last_status_control_state==cycle->control_state)return false;
    s_status_signature_valid=1u;s_last_status_fault_flags=cycle->fault_flags;
    s_last_status_offline_mask=cycle->offline_port_mask;
    s_last_status_reject=cycle->last_reject_code;
    s_last_status_control_state=cycle->control_state;return true;
}

static __attribute__((noinline)) bool status_signature_changed(void)
{
    const control_cycle_snapshot_t cycle=cycle_snapshot();
    return status_changed(&cycle);
}

static void publish_device(esd_link_qos_t qos)
{
    const control_cycle_snapshot_t cycle=cycle_snapshot();
    uint8_t p[ESD_LINK_V1_DEVICE_STATUS_PAYLOAD_LEN]={0};
    esd_link_stats_t st;esd_link_get_stats(&s_link,&st);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_SESSION_OFFSET],cycle.session_id);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_LOWER_BOOT_ID_OFFSET],s_cfg.lower_boot_id);
    p[ESD_LINK_V1_DEVICE_STATUS_CONTROL_STATE_OFFSET]=cycle.control_state;
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_FAULT_FLAGS_OFFSET],cycle.fault_flags);
    esd_link_v1_le16_write(&p[ESD_LINK_V1_DEVICE_STATUS_LAST_REJECT_CODE_OFFSET],cycle.last_reject_code);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_VALID_COMMAND_COUNT_OFFSET],cycle.valid_command_count);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_INVALID_FRAME_COUNT_OFFSET],st.cobs_errors+st.crc_errors+st.length_errors);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_REJECTED_COMMAND_COUNT_OFFSET],cycle.rejected_command_count);
    esd_link_v1_le16_write(&p[ESD_LINK_V1_DEVICE_STATUS_COMMAND_AGE_MS_OFFSET],cycle.command_age_ms);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_ACTIVE_PORT_MASK_OFFSET],cycle.active_port_mask);
    esd_link_v1_le32_write(&p[ESD_LINK_V1_DEVICE_STATUS_OFFLINE_PORT_MASK_OFFSET],cycle.offline_port_mask);
    (void)esd_link_publish_qos(&s_link,qos,ESD_LINK_SERVICE_STATE,
        ESD_LINK_STATE_DEVICE_STATUS,0u,++s_event_sequence,p,sizeof(p));
}

static void service_link(void)
{
    const bool connected=s_cfg.transport.is_connected(s_cfg.transport.user);
    if (s_connected && !connected) {
        if(s_link_disconnect_generation==s_disconnect_generation)TSKPTT_ESDLink_NotifyDisconnect();
        s_link_disconnect_generation=s_disconnect_generation;s_hskp_period_ms=0u;
        if(s_cfg.hskp_config)s_cfg.hskp_config(s_cfg.project_user,false,0u);
    }
    s_connected=connected?1u:0u;esd_link_set_transport_connected(&s_link,connected);
    bool success;while(s_cfg.transport.tx_take_complete(s_cfg.transport.user,&success))esd_link_on_tx_complete(&s_link,success);
    for(;;){size_t n=s_cfg.transport.rx_read(s_cfg.transport.user,s_cfg.rx_storage,s_cfg.rx_storage_size);if(n==0u)break;(void)esd_link_feed_rx(&s_link,s_cfg.rx_storage,n);}
    uint32_t overflow=s_cfg.transport.rx_overflow_bytes(s_cfg.transport.user);esd_link_record_rx_overflow(&s_link,overflow-s_last_rx_overflow);s_last_rx_overflow=overflow;
    config_response_t r;while(osMessageQueueGet(s_response_queue,&r,NULL,0u)==osOK)(void)esd_link_send_response(&s_link,r.service,r.message,r.sequence,r.status,r.flags,r.body,r.body_len);
    if(connected)(void)esd_link_service_tx(&s_link);
}

static __attribute__((noinline)) void publish_debug_snapshot(void)
{
    if(s_cfg.snapshot==NULL)return;
    esd_link_stats_t st;esd_link_get_stats(&s_link,&st);
    s_cfg.snapshot(s_cfg.project_user,&st,
        (tskptt_esd_control_state_t)s_control_state,
        s_control_deadline_miss,s_disconnect_generation,
        uxTaskGetStackHighWaterMark(s_link_task),
        s_ctrl_task?uxTaskGetStackHighWaterMark(s_ctrl_task):0u,
        s_config_task?uxTaskGetStackHighWaterMark(s_config_task):0u);
}

esd_link_result_t TSKPTT_ESDLink_Configure(const tskptt_esd_link_config_t *cfg)
{
    if ((cfg==NULL)||s_configured||(cfg->now_ms==NULL)||(cfg->tx_storage==NULL)||
        (((uintptr_t)cfg->tx_storage&31u)!=0u)||(cfg->tx_storage_size<ESD_LINK_TX_SLOT_COUNT*ESD_LINK_V1_FRAME_SLOT_SIZE)||
        (cfg->rx_storage==NULL)||(cfg->rx_storage_size==0u)||(cfg->lower_boot_id==0u)||
        (cfg->transport.init==NULL)||(cfg->transport.start_rx==NULL)||(cfg->transport.rx_read==NULL)||
        (cfg->transport.rx_overflow_bytes==NULL)||(cfg->transport.tx_submit==NULL)||
        (cfg->transport.tx_take_complete==NULL)||(cfg->transport.is_connected==NULL)||
        ((cfg->schema_id!=1u)&&(cfg->schema_id!=2u))||
        ((cfg->schema_id==2u)&&(cfg->port_count>11u))) return ESD_LINK_ERR_PARAM;
    const uint16_t control_rate=(cfg->control_rate_hz!=0u)?cfg->control_rate_hz:TSKPTT_ESD_LINK_DEFAULT_CONTROL_RATE_HZ;
    if((control_rate>1000u)||((1000u%control_rate)!=0u))return ESD_LINK_ERR_PARAM;
    s_cfg=*cfg;s_control_rate_hz=control_rate;s_control_period_ms=1000u/control_rate;
    if(!esd_port_registry_init(&s_ports,cfg->ports,cfg->port_count,control_rate,(uint16_t)((1u<<1)|(1u<<2))))return ESD_LINK_ERR_PARAM;
    for(size_t i=0u;i<s_ports.count;++i){const uint8_t id=s_ports.ports[i].port_id;s_zero[id]=cfg->initial_zero_offsets?cfg->initial_zero_offsets[id]:s_ports.ports[i].software_zero_rad;}
    refresh_fingerprint();s_control_state=(cfg->boot_storage_healthy&&cfg->configuration_healthy)?TSKPTT_ESD_CONTROL_NO_SESSION:TSKPTT_ESD_CONTROL_FAULT_LATCHED;
    memset(&s_cycle,0,sizeof(s_cycle));s_cycle.active_port_mask=active_mask();s_cycle.control_state=s_control_state;
    if(!cfg->configuration_healthy)s_cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_CONFIGURATION;
    if(!cfg->boot_storage_healthy||!cfg->zero_storage_healthy)s_cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_PERSISTENCE;
    s_configured=1u;return ESD_LINK_OK;
}

bool TSKPTT_ESDLink_InitResources(void)
{
    if(!s_configured)return false;
    if(s_cfg_queue==NULL)s_cfg_queue=osMessageQueueNew(CFG_QUEUE_DEPTH,sizeof(config_work_t),NULL);
    if(s_response_queue==NULL)s_response_queue=osMessageQueueNew(CFG_RESPONSE_DEPTH,sizeof(config_response_t),NULL);
    return s_cfg_queue&&s_response_queue;
}

void TSKPTT_ESDLink_LinkTask(void *argument)
{
    (void)argument;s_link_task=xTaskGetCurrentTaskHandle();configASSERT(TSKPTT_ESDLink_InitResources());configASSERT(start_link()==ESD_LINK_OK);
    uint32_t device_at=now_ms(),snapshot_at=device_at;
    for(;;){uint32_t note=0u;(void)xTaskNotifyWait(0u,UINT32_MAX,&note,pdMS_TO_TICKS(2u));service_link();uint32_t now=now_ms();
        if(status_signature_changed())publish_device(ESD_LINK_QOS_REALTIME);
        if((note&NOTIFY_STATE)!=0u){publish_state(now);if(s_connected)(void)esd_link_service_tx(&s_link);}
        if((now-device_at)>=DEVICE_PERIOD_MS){device_at=now;publish_device(ESD_LINK_QOS_DIAGNOSTIC);}
        if((now-snapshot_at)>=1000u){snapshot_at=now;publish_debug_snapshot();}
    }
}

static void control_once(void)
{
    uint32_t start=cycles(),now=now_ms();
        if(s_ctrl_disconnect_generation!=s_disconnect_generation){s_ctrl_disconnect_generation=s_disconnect_generation;disconnect_safe_path(now);}
        feedback_snapshot_t next;memset(&next,0,sizeof(next));next.count=(uint8_t)s_ports.count;next.sample_sequence=++s_state_sequence;next.sample_time_us=now*1000u;
        const bool imu_ok=s_cfg.imu_snapshot&&s_cfg.imu_snapshot(s_cfg.project_user,&next.sample_time_us,&next.imu_valid,next.accel,next.gyro,next.quat)&&(next.imu_valid!=0u);
        uint32_t offline_mask=0u;bool actuator_fault=false;
        for(size_t i=0u;i<s_ports.count;++i){const esd_port_config_t *p=&s_ports.ports[i];esd_port_feedback_t f={0};const bool sampled=p->ops->sample(p->user,now,&f);
            if(!sampled){esd_port_feedback_t last={0};uint8_t previous_count;
                taskENTER_CRITICAL();previous_count=s_cycle.feedback.count;if(i<previous_count)last=s_cycle.feedback.feedback[i];taskEXIT_CRITICAL();
                const uint32_t age=now-last.sample_ms;
                if(i<previous_count&&last.valid&&last.online&&(age>0x80000000u||age<=p->watchdog_ms)){f=last;if(f.fault)actuator_fault=true;next.feedback[i]=f;continue;}}
            if(!sampled||!f.valid||!f.online){offline_mask|=1u<<p->port_id;f.valid=0u;f.online=0u;}
            else if(f.fault){actuator_fault=true;}
            if(f.valid){f.position_rad=(float)p->direction*f.position_rad/p->gear_ratio-s_zero[p->port_id];f.velocity_rad_s=(float)p->direction*f.velocity_rad_s/p->gear_ratio;f.effort_nm=(float)p->direction*(f.effort_nm/p->effort_to_backend)*p->gear_ratio*p->motor_efficiency;
                if(!isfinite(f.position_rad)||!isfinite(f.velocity_rad_s)||!isfinite(f.effort_nm)){offline_mask|=1u<<p->port_id;f.valid=0u;f.online=0u;}}
            next.feedback[i]=f;}
        command_snapshot_t cmd;taskENTER_CRITICAL();cmd=s_command;taskEXIT_CRITICAL();
        uint32_t timeout_ms=200u;
        if(cmd.valid){for(size_t i=0u;i<s_ports.count;++i)if(s_ports.ports[i].watchdog_ms<timeout_ms)timeout_ms=s_ports.ports[i].watchdog_ms;}
        const uint32_t command_age=now-(cmd.valid?cmd.received_ms:s_last_command_ms);
        if(s_control_deadline_miss!=s_seen_deadline_miss){s_seen_deadline_miss=s_control_deadline_miss;s_control_fault_latched=1u;}
        if(actuator_fault)s_control_fault_latched=1u;
        if(s_output_gate&&(!imu_ok||offline_mask!=0u||actuator_fault||s_control_fault_latched)){s_last_reject=(offline_mask!=0u)?ESD_LINK_STATUS_DEVICE_OFFLINE:ESD_LINK_STATUS_INVALID_STATE;emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);}
        else if(s_output_gate&&s_control_state==TSKPTT_ESD_CONTROL_SAFE_DAMPING){cmd.valid=0u;if(!run_safe_damping(now,&next)){s_last_reject=ESD_LINK_STATUS_INTERNAL_ERROR;s_control_fault_latched=1u;emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);}}
        else if(s_output_gate&&cmd.valid&&(command_age<=timeout_ms)){for(size_t i=0u;i<s_ports.count;++i){const esd_port_config_t *p=&s_ports.ports[i];if(!esd_port_apply_command(p,&next.feedback[i],&cmd.command[i],s_zero[p->port_id])){s_control_fault_latched=1u;break;}}if(s_control_fault_latched){s_last_reject=ESD_LINK_STATUS_INTERNAL_ERROR;emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);}else s_control_state=TSKPTT_ESD_CONTROL_ACTIVE;}
        else if(s_output_gate&&(command_age>timeout_ms)){s_last_reject=ESD_LINK_STATUS_INVALID_STATE;begin_safe_damping(now,TSKPTT_ESD_CONTROL_SAFE_DAMPING);cmd.valid=0u;if(!run_safe_damping(now,&next)){s_last_reject=ESD_LINK_STATUS_INTERNAL_ERROR;s_control_fault_latched=1u;emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);}}
        control_cycle_snapshot_t cycle;memset(&cycle,0,sizeof(cycle));cycle.feedback=next;cycle.command=cmd;cycle.session_id=s_session;cycle.active_port_mask=active_mask();cycle.offline_port_mask=offline_mask;cycle.valid_command_count=s_valid_commands;cycle.rejected_command_count=s_rejected_commands;cycle.last_reject_code=s_last_reject;cycle.control_state=s_control_state;cycle.output_gate=s_output_gate;
        uint32_t age=now-(cmd.valid?cmd.received_ms:s_last_command_ms);cycle.command_age_ms=(uint16_t)(age>65535u?65535u:age);
        if(s_control_state==TSKPTT_ESD_CONTROL_SAFE_DAMPING&&s_safe_damping_terminal_state!=TSKPTT_ESD_CONTROL_DISABLED)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_COMMAND_WATCHDOG;
        if(!imu_ok)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_IMU;
        if(offline_mask!=0u)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_ACTUATOR_OFFLINE;
        if(!s_cfg.configuration_healthy)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_CONFIGURATION;
        if(s_control_fault_latched||actuator_fault)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_CONTROL;
        if(!s_cfg.boot_storage_healthy||!s_cfg.zero_storage_healthy)cycle.fault_flags|=ESD_LINK_DEVICE_FAULT_PERSISTENCE;
        taskENTER_CRITICAL();s_cycle=cycle;taskEXIT_CRITICAL();
        if(s_link_task!=NULL)(void)xTaskNotify(s_link_task,NOTIFY_STATE,eSetBits);
        if(s_cfg.cycles&&((cycles()-start)>0u)){/* target timing is exposed through deadline counter */}
}

void TSKPTT_ESDLink_ControlTask(void *argument)
{
    (void)argument;s_ctrl_task=xTaskGetCurrentTaskHandle();TickType_t wake=xTaskGetTickCount();
    const TickType_t period=pdMS_TO_TICKS(s_control_period_ms);
    for(;;){vTaskDelayUntil(&wake,period);control_once();if((int32_t)(xTaskGetTickCount()-wake)>0)s_control_deadline_miss++;
    }
}

static void cache_completion(const config_work_t *w,const config_response_t *r)
{
    uint32_t session=esd_link_v1_le32_read(w->payload),tx=esd_link_v1_le32_read(&w->payload[4]);
    taskENTER_CRITICAL();transaction_entry_t *e=transaction_find(w->service,w->message,session,tx);if(e){e->pending=0u;e->status=r->status;e->body_len=(uint8_t)r->body_len;copy_bytes(e->body,r->body,r->body_len);}taskEXIT_CRITICAL();
}

static void run_enable(const config_work_t *w,config_response_t *r)
{
    const uint32_t session=esd_link_v1_le32_read(w->payload),tx=esd_link_v1_le32_read(&w->payload[4]);
    uint32_t requested=0u,affected=0u,verified=0u,rollback=0u;uint8_t failed=0xFFu;uint16_t detail=0u;
    r->status=ESD_LINK_STATUS_OK;
    if((w->length<10u)||(w->payload[8]!=1u)||(w->payload[9]!=s_ports.count)||(w->length!=(uint16_t)(10u+w->payload[9])))r->status=ESD_LINK_STATUS_INVALID_LENGTH;
    for(size_t i=0u;(r->status==0u)&&(i<w->payload[9]);++i){uint8_t id=w->payload[10u+i];const esd_port_config_t *p=esd_port_find(&s_ports,id);if(!p||(requested&(1u<<id))){r->status=p?ESD_LINK_STATUS_DUPLICATE_PORT_ID:ESD_LINK_STATUS_INVALID_PORT_ID;failed=id;break;}requested|=1u<<id;}
    if((r->status==0u)&&((requested!=active_mask())||(s_ports.count==0u)))r->status=ESD_LINK_STATUS_INVALID_STATE;
    if((r->status==0u)&&((s_control_state!=TSKPTT_ESD_CONTROL_DISABLED)||
       !s_cfg.boot_storage_healthy||!s_cfg.zero_storage_healthy||!s_cfg.configuration_healthy||s_control_fault_latched))r->status=ESD_LINK_STATUS_INVALID_STATE;
    feedback_snapshot_t current;taskENTER_CRITICAL();current=s_cycle.feedback;taskEXIT_CRITICAL();if((r->status==0u)&&(current.imu_valid==0u))r->status=ESD_LINK_STATUS_INVALID_STATE;
    uint32_t command_ms=now_ms(),baseline[ESD_PORT_ACTIVE_MAX]={0};
    for(size_t i=0u;(r->status==0u)&&(i<s_ports.count);++i){const esd_port_config_t *p=&s_ports.ports[i];esd_port_feedback_t f;if(!p->ops->sample(p->user,command_ms,&f)||!f.valid||!f.online||f.fault||!p->ops->request_enable(p->user)){r->status=ESD_LINK_STATUS_DEVICE_OFFLINE;failed=p->port_id;break;}baseline[i]=f.sample_ms;affected|=1u<<p->port_id;bool probe_ok;if(p->confirm_mode==ESD_PORT_CONFIRM_LOCAL_GATE)probe_ok=p->ops->set_motor_effort(p->user,0.0f);else {float dq=(float)p->direction*f.velocity_rad_s/p->gear_ratio;float tau=-p->safe_damping_nm_s_rad*dq;if(tau>p->effort_max_nm)tau=p->effort_max_nm;if(tau<-p->effort_max_nm)tau=-p->effort_max_nm;probe_ok=esd_port_apply_effort(p,tau);}if(!probe_ok){r->status=ESD_LINK_STATUS_VERIFY_FAILED;failed=p->port_id;break;}}
    for(uint32_t elapsed=0u;(r->status==0u)&&(elapsed<ENABLE_VERIFY_MS);++elapsed){verified=0u;for(size_t i=0u;i<s_ports.count;++i){const esd_port_config_t *p=&s_ports.ports[i];esd_port_feedback_t f;if(p->ops->sample(p->user,now_ms(),&f)&&f.valid&&!f.fault&&((int32_t)(f.sample_ms-baseline[i])>0)&&p->ops->verify_enabled(p->user,command_ms,&f))verified|=1u<<p->port_id;}if(verified==requested)break;osDelay(1u);}
    if((r->status==0u)&&(verified!=requested)){r->status=ESD_LINK_STATUS_VERIFY_FAILED;detail=1u;}
    if(r->status==0u){for(size_t i=0u;i<s_ports.count;++i)if(s_ports.ports[i].confirm_mode>detail)detail=s_ports.ports[i].confirm_mode;taskENTER_CRITICAL();s_output_gate=1u;s_control_state=TSKPTT_ESD_CONTROL_ENABLED_WAIT_COMMAND;s_last_command_ms=now_ms();taskEXIT_CRITICAL();}
    else {emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);for(size_t i=0u;i<s_ports.count;++i){const esd_port_config_t *p=&s_ports.ports[i];if((affected&(1u<<p->port_id))&&p->ops->verify_disabled(p->user))rollback|=1u<<p->port_id;}}
    response_body(r->body,session,tx,failed,detail,requested,affected,verified,rollback);r->body_len=CONFIG_RESPONSE_LEN;
}

static void run_zero(const config_work_t *w,config_response_t *r)
{
    const uint32_t session=esd_link_v1_le32_read(w->payload),tx=esd_link_v1_le32_read(&w->payload[4]);
    uint32_t requested=0u,affected=0u,verified=0u,rollback=0u;uint8_t failed=0xFFu;r->status=ESD_LINK_STATUS_OK;
    if((w->length<15u)||(w->payload[8]>1u)||(w->payload[9]==0u)||(w->payload[9]>ESD_PORT_ACTIVE_MAX)||
       (w->length!=(uint16_t)(10u+5u*w->payload[9])))r->status=ESD_LINK_STATUS_INVALID_LENGTH;
    else if(s_control_state!=TSKPTT_ESD_CONTROL_DISABLED)r->status=ESD_LINK_STATUS_INVALID_STATE;
    float old[ESD_PORT_ID_COUNT],candidate[ESD_PORT_ID_COUNT],assigned_by_id[ESD_PORT_ID_COUNT]={0};taskENTER_CRITICAL();memcpy(old,s_zero,sizeof(old));taskEXIT_CRITICAL();memcpy(candidate,old,sizeof(candidate));
    feedback_snapshot_t snap;taskENTER_CRITICAL();snap=s_cycle.feedback;taskEXIT_CRITICAL();
    for(size_t i=0u;(r->status==0u)&&(i<w->payload[9]);++i){const uint8_t *p=&w->payload[10u+5u*i];uint8_t id=p[0];float assigned;const esd_port_config_t *port=esd_port_find(&s_ports,id);
        if(!port){r->status=ESD_LINK_STATUS_INVALID_PORT_ID;failed=id;break;}if(requested&(1u<<id)){r->status=ESD_LINK_STATUS_DUPLICATE_PORT_ID;failed=id;break;}if(!get_f32(&p[1],&assigned)){r->status=ESD_LINK_STATUS_INVALID_VALUE;failed=id;break;}
        if((port->capabilities&ESD_PORT_CAP_SOFTWARE_ZERO)==0u){r->status=ESD_LINK_STATUS_UNSUPPORTED_CAPABILITY;failed=id;break;}
        if((w->payload[8]==0u)&&((port->capabilities&ESD_PORT_CAP_VOLATILE_ZERO)==0u)){r->status=ESD_LINK_STATUS_UNSUPPORTED_CAPABILITY;failed=id;break;}
        size_t index=0u;while(index<s_ports.count&&s_ports.ports[index].port_id!=id)index++;if(index==s_ports.count||!snap.feedback[index].valid||!snap.feedback[index].online||snap.feedback[index].fault){r->status=ESD_LINK_STATUS_DEVICE_OFFLINE;failed=id;break;}
        requested|=1u<<id;assigned_by_id[id]=assigned;candidate[id]=old[id]+snap.feedback[index].position_rad-assigned;affected|=1u<<id;}
    if((r->status==0u)&&(w->payload[8]!=0u)&&((s_cfg.persist_zero==NULL)||!s_cfg.persist_zero(s_cfg.project_user,candidate,requested)))r->status=ESD_LINK_STATUS_PERSIST_FAILED;
    if(r->status==0u){
        taskENTER_CRITICAL();memcpy(s_zero,candidate,sizeof(candidate));taskEXIT_CRITICAL();
#ifdef TSKPTT_ESD_LINK_TESTING
        control_once();
#else
        osDelay(s_control_period_ms+1u);
#endif
        feedback_snapshot_t check;taskENTER_CRITICAL();check=s_cycle.feedback;taskEXIT_CRITICAL();
        for(size_t i=0u;i<s_ports.count;++i){const uint8_t id=s_ports.ports[i].port_id;if((requested&(1u<<id))==0u)continue;
            if(!check.feedback[i].valid||!check.feedback[i].online||check.feedback[i].fault||
               fabsf(check.feedback[i].position_rad-assigned_by_id[id])>0.001f){r->status=ESD_LINK_STATUS_VERIFY_FAILED;failed=id;break;}
            verified|=1u<<id;}
        if(r->status!=0u){taskENTER_CRITICAL();memcpy(s_zero,old,sizeof(old));taskEXIT_CRITICAL();rollback=affected;
            if((w->payload[8]!=0u)&&s_cfg.persist_zero&&
               !s_cfg.persist_zero(s_cfg.project_user,old,requested))r->status=ESD_LINK_STATUS_PERSIST_FAILED;}
    }
    if(r->status!=0u){refresh_fingerprint();emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);}
    else {if(w->payload[8]!=0u)s_cfg.zero_storage_healthy=true;refresh_fingerprint();invalidate_session(TSKPTT_ESD_CONTROL_NO_SESSION);}
    response_body(r->body,session,tx,failed,0u,requested,affected,verified,rollback);r->body_len=CONFIG_RESPONSE_LEN;
}

static uint32_t staging_complete_mask(uint8_t count)
{
    return (count == 32u) ? UINT32_MAX : ((1u << count) - 1u);
}

static void port_config_response(const config_work_t *w,config_response_t *r,
                                 uint16_t status,uint8_t failed,
                                 uint16_t detail,uint32_t requested,
                                 uint32_t affected,uint32_t verified,
                                 uint32_t rollback)
{
    r->status=status;
    response_body(r->body,esd_link_v1_le32_read(w->payload),
                  esd_link_v1_le32_read(&w->payload[4]),failed,detail,
                  requested,affected,verified,rollback);
    r->body_len=CONFIG_RESPONSE_LEN;
}

static void expire_port_staging(void)
{
    if(s_port_staging.active&&
       ((uint32_t)(now_ms()-s_port_staging.last_activity_ms)>
        PORT_CONFIG_TIMEOUT_MS))memset(&s_port_staging,0,sizeof(s_port_staging));
}

static void run_port_config_begin(const config_work_t *w,config_response_t *r)
{
    expire_port_staging();
    
    if(w->length!=ESD_LINK_V1_PORT_CONFIG_BEGIN_PAYLOAD_LEN){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,0u,0u,0u,0u);return;}
    if(s_control_state!=TSKPTT_ESD_CONTROL_DISABLED||s_output_gate){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_STATE,0xFFu,0u,0u,0u,0u,0u);return;}
    const uint8_t format=w->payload[ESD_LINK_V1_PORT_CONFIG_BEGIN_FORMAT_OFFSET];
    const uint8_t entry_size=w->payload[ESD_LINK_V1_PORT_CONFIG_BEGIN_ENTRY_SIZE_OFFSET];
    const uint8_t count=w->payload[ESD_LINK_V1_PORT_CONFIG_BEGIN_COUNT_OFFSET];
    const uint32_t config_id=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_ID_OFFSET]);
    const uint32_t table_crc=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_BEGIN_TABLE_CRC_OFFSET]);
    if(format!=ESD_PORT_CONFIG_FORMAT_VERSION||
       entry_size!=ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN||count==0u||
       count>ESD_PORT_ACTIVE_MAX||config_id==0u||
       w->payload[ESD_LINK_V1_PORT_CONFIG_BEGIN_RESERVED_OFFSET]!=0u){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_VALUE,0xFFu,0u,0u,0u,0u,0u);return;}
    if(s_port_staging.active&&s_port_staging.config_id==config_id&&
       s_port_staging.count==count&&s_port_staging.table_crc==table_crc){
        s_port_staging.last_activity_ms=now_ms();
    }else{
        memset(&s_port_staging,0,sizeof(s_port_staging));
        s_port_staging.active=1u;s_port_staging.config_id=config_id;
        s_port_staging.count=count;s_port_staging.table_crc=table_crc;
        s_port_staging.last_activity_ms=now_ms();
    }
    port_config_response(w,r,ESD_LINK_STATUS_OK,0xFFu,count,
                         staging_complete_mask(count),0u,0u,0u);
}

static void run_port_config_chunk(const config_work_t *w,config_response_t *r)
{
    expire_port_staging();
    if(w->length<ESD_LINK_V1_PORT_CONFIG_CHUNK_PREFIX_LEN){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,0u,0u,0u,0u);return;}
    const uint32_t config_id=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_ID_OFFSET]);
    const uint8_t start=w->payload[ESD_LINK_V1_PORT_CONFIG_CHUNK_START_OFFSET];
    const uint8_t count=w->payload[ESD_LINK_V1_PORT_CONFIG_CHUNK_COUNT_OFFSET];
    const size_t expected=ESD_LINK_V1_PORT_CONFIG_CHUNK_PREFIX_LEN+
        (size_t)count*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN;
    if(count==0u||w->length!=expected||
       w->payload[ESD_LINK_V1_PORT_CONFIG_CHUNK_RESERVED_OFFSET]!=0u||
       w->payload[ESD_LINK_V1_PORT_CONFIG_CHUNK_RESERVED_OFFSET+1u]!=0u){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,0u,0u,0u,0u);return;}
    if(s_control_state!=TSKPTT_ESD_CONTROL_DISABLED||s_output_gate||
       !s_port_staging.active||config_id!=s_port_staging.config_id){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_STATE,0xFFu,0u,0u,0u,0u,0u);return;}
    if((size_t)start+count>s_port_staging.count){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_VALUE,start,0u,0u,0u,0u,0u);return;}
    for(uint8_t i=0u;i<count;++i){
        const uint8_t index=(uint8_t)(start+i);
        const uint8_t *source=&w->payload[ESD_LINK_V1_PORT_CONFIG_CHUNK_ENTRIES_OFFSET+
            (size_t)i*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN];
        const uint32_t bit=1u<<index;
        if((s_port_staging.received_mask&bit)!=0u&&
           memcmp(s_port_staging.raw[index],source,
                  ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN)!=0){
            port_config_response(w,r,ESD_LINK_STATUS_CONFLICT,index,0u,
                                 s_port_staging.received_mask,0u,0u,0u);return;}
        /* raw may be unaligned; keep stores byte-wide with UNALIGN_TRP enabled. */
        copy_bytes(s_port_staging.raw[index],source,
                   ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN);
        s_port_staging.received_mask|=bit;
    }
    s_port_staging.last_activity_ms=now_ms();
    port_config_response(w,r,ESD_LINK_STATUS_OK,0xFFu,count,
                         staging_complete_mask(s_port_staging.count),
                         s_port_staging.received_mask,0u,0u);
}

static bool decode_port_declaration(const uint8_t *p,
                                    esd_port_declaration_t *d)
{
    memset(d,0,sizeof(*d));
    d->port_id=p[ESD_LINK_V1_PORT_CONFIG_ENTRY_PORT_OFFSET];
    d->device=p[ESD_LINK_V1_PORT_CONFIG_ENTRY_DEVICE_OFFSET];
    d->logical_can=p[ESD_LINK_V1_PORT_CONFIG_ENTRY_LOGICAL_CAN_OFFSET];
    return p[ESD_LINK_V1_PORT_CONFIG_ENTRY_RESERVED_OFFSET]==0u&&
           get_f32(&p[ESD_LINK_V1_PORT_CONFIG_ENTRY_VELOCITY_MAX_OFFSET],
                   &d->velocity_max_rad_s)&&
           get_f32(&p[ESD_LINK_V1_PORT_CONFIG_ENTRY_EFFORT_MAX_OFFSET],
                   &d->effort_max_nm);
}

static void suspend_runtime_tasks(void)
{
#ifndef TSKPTT_ESD_LINK_TESTING
    if(s_ctrl_task!=NULL)vTaskSuspend(s_ctrl_task);
    if(s_link_task!=NULL)vTaskSuspend(s_link_task);
#endif
}

static void resume_runtime_tasks(void)
{
#ifndef TSKPTT_ESD_LINK_TESTING
    if(s_link_task!=NULL)vTaskResume(s_link_task);
    if(s_ctrl_task!=NULL)vTaskResume(s_ctrl_task);
#endif
}

static void latch_configuration_lost(void)
{
    memset(&s_ports,0,sizeof(s_ports));
    s_cfg.ports=NULL;
    s_cfg.port_count=0u;
    s_cfg.configuration_healthy=false;
    memset(&s_command,0,sizeof(s_command));
    memset(&s_cycle,0,sizeof(s_cycle));
    s_cycle.control_state=TSKPTT_ESD_CONTROL_FAULT_LATCHED;
    s_cycle.fault_flags=ESD_LINK_DEVICE_FAULT_CONFIGURATION;
    refresh_fingerprint();
    s_control_fault_latched=1u;
    emergency_disable(TSKPTT_ESD_CONTROL_FAULT_LATCHED);
    invalidate_session(TSKPTT_ESD_CONTROL_FAULT_LATCHED);
}

static void run_port_config_commit(const config_work_t *w,config_response_t *r)
{
    expire_port_staging();
    if(w->length!=ESD_LINK_V1_PORT_CONFIG_COMMIT_PAYLOAD_LEN){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,0u,0u,0u,0u);return;}
    const uint32_t config_id=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_ID_OFFSET]);
    const uint32_t table_crc=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_COMMIT_TABLE_CRC_OFFSET]);
    if(s_control_state!=TSKPTT_ESD_CONTROL_DISABLED||s_output_gate||
       !s_port_staging.active||config_id!=s_port_staging.config_id){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_STATE,0xFFu,0u,0u,0u,0u,0u);return;}
    const uint32_t complete=staging_complete_mask(s_port_staging.count);
    if(s_port_staging.received_mask!=complete){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_STATE,0xFFu,1u,
                             complete,s_port_staging.received_mask,0u,0u);return;}
    const uint32_t actual_crc=esd_link_v1_crc32(&s_port_staging.raw[0][0],
        (size_t)s_port_staging.count*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN);
    if(table_crc!=s_port_staging.table_crc||actual_crc!=table_crc){
        port_config_response(w,r,ESD_LINK_STATUS_CONFIG_MISMATCH,0xFFu,2u,
                             complete,0u,0u,0u);return;}
    esd_port_declaration_t declarations[ESD_PORT_ACTIVE_MAX];
    for(uint8_t i=0u;i<s_port_staging.count;++i){
        if(!decode_port_declaration(s_port_staging.raw[i],&declarations[i])){
            port_config_response(w,r,ESD_LINK_STATUS_INVALID_VALUE,i,0u,
                                 complete,0u,0u,0u);return;}
    }
    if(s_cfg.apply_port_config==NULL){
        port_config_response(w,r,ESD_LINK_STATUS_UNSUPPORTED_CAPABILITY,0xFFu,0u,
                             complete,0u,0u,0u);return;}
    if(s_cfg.schema_id==2u&&s_port_staging.count>11u){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_VALUE,0xFFu,3u,
                             complete,0u,0u,0u);return;}

    tskptt_esd_port_config_result_t result;memset(&result,0,sizeof(result));
    suspend_runtime_tasks();
    uint16_t status=s_cfg.apply_port_config(s_cfg.project_user,declarations,
        s_port_staging.count,s_zero,&result);
    if(result.configuration_lost){
        latch_configuration_lost();
        status=ESD_LINK_STATUS_INTERNAL_ERROR;
    }else if(status==ESD_LINK_STATUS_OK&&!result.unchanged){
        esd_port_registry_t next;
        if(!esd_port_registry_init(&next,result.ports,result.port_count,
               s_control_rate_hz,(uint16_t)((1u<<1)|(1u<<2)))){
            latch_configuration_lost();
            status=ESD_LINK_STATUS_INTERNAL_ERROR;
        }else{
            s_ports=next;s_cfg.ports=result.ports;s_cfg.port_count=result.port_count;
            memcpy(s_zero,result.zero_offsets,sizeof(s_zero));
            s_cfg.zero_storage_healthy=result.zero_storage_healthy;
            s_cfg.configuration_healthy=true;
            memset(&s_command,0,sizeof(s_command));memset(&s_cycle,0,sizeof(s_cycle));
            s_cycle.active_port_mask=active_mask();
            refresh_fingerprint();
            invalidate_session(TSKPTT_ESD_CONTROL_NO_SESSION);
            s_cycle.control_state=TSKPTT_ESD_CONTROL_NO_SESSION;
        }
    }
    resume_runtime_tasks();

    if(status==ESD_LINK_STATUS_OK){
        const uint32_t mask=active_mask();
        memset(&s_port_staging,0,sizeof(s_port_staging));
        port_config_response(w,r,status,0xFFu,result.unchanged?1u:0u,
                             mask,mask,mask,0u);
    }else{
        port_config_response(w,r,status,0xFFu,0u,complete,0u,0u,0u);
    }
}

static void run_port_config_update(const config_work_t *w,config_response_t *r)
{
    if(w->length<ESD_LINK_V1_PORT_CONFIG_UPDATE_PREFIX_LEN){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,
                             0u,0u,0u,0u);return;}
    if(s_control_state!=TSKPTT_ESD_CONTROL_DISABLED||s_output_gate){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_STATE,0xFFu,0u,
                             0u,0u,0u,0u);return;}

    const uint8_t format=
        w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_FORMAT_OFFSET];
    const uint8_t entry_size=
        w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_ENTRY_SIZE_OFFSET];
    const uint8_t count=
        w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_COUNT_OFFSET];
    const size_t expected=ESD_LINK_V1_PORT_CONFIG_UPDATE_PREFIX_LEN+
        (size_t)count*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN;
    if(w->length!=expected){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,
                             0u,0u,0u,0u);return;}
    if(format!=ESD_PORT_CONFIG_FORMAT_VERSION||
       entry_size!=ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN||count==0u||
       count>ESD_PORT_ACTIVE_MAX||
       w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_RESERVED_OFFSET]!=0u){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_VALUE,0xFFu,0u,
                             0u,0u,0u,0u);return;}

    const uint8_t *entries=
        &w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_ENTRIES_OFFSET];
    const uint32_t table_crc=esd_link_v1_le32_read(
        &w->payload[ESD_LINK_V1_PORT_CONFIG_UPDATE_TABLE_CRC_OFFSET]);
    const uint32_t actual_crc=esd_link_v1_crc32(
        entries,(size_t)count*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN);
    if(actual_crc!=table_crc){
        port_config_response(w,r,ESD_LINK_STATUS_CONFIG_MISMATCH,0xFFu,2u,
                             staging_complete_mask(count),0u,0u,0u);return;}

    memset(&s_port_staging,0,sizeof(s_port_staging));
    s_port_staging.active=1u;
    s_port_staging.config_id=esd_link_v1_le32_read(&w->payload[4]);
    s_port_staging.count=count;
    s_port_staging.table_crc=table_crc;
    s_port_staging.received_mask=staging_complete_mask(count);
    s_port_staging.last_activity_ms=now_ms();
    /* raw may be unaligned; keep stores byte-wide with UNALIGN_TRP enabled. */
    copy_bytes(&s_port_staging.raw[0][0],entries,
               (size_t)count*ESD_LINK_V1_PORT_CONFIG_ENTRY_LEN);

    config_work_t commit=*w;
    commit.message=ESD_LINK_CONFIG_PORT_CONFIG_COMMIT;
    commit.length=ESD_LINK_V1_PORT_CONFIG_COMMIT_PAYLOAD_LEN;
    esd_link_v1_le32_write(&commit.payload[ESD_LINK_V1_PORT_CONFIG_ID_OFFSET],
                           s_port_staging.config_id);
    esd_link_v1_le32_write(
        &commit.payload[ESD_LINK_V1_PORT_CONFIG_COMMIT_TABLE_CRC_OFFSET],
        table_crc);
    run_port_config_commit(&commit,r);
    memset(&s_port_staging,0,sizeof(s_port_staging));
}

static void run_port_config_abort(const config_work_t *w,config_response_t *r)
{
    if(w->length!=ESD_LINK_V1_PORT_CONFIG_ABORT_PAYLOAD_LEN){
        port_config_response(w,r,ESD_LINK_STATUS_INVALID_LENGTH,0xFFu,0u,0u,0u,0u,0u);return;}
    const uint32_t config_id=esd_link_v1_le32_read(&w->payload[ESD_LINK_V1_PORT_CONFIG_ID_OFFSET]);
    if(s_port_staging.active&&s_port_staging.config_id==config_id)
        memset(&s_port_staging,0,sizeof(s_port_staging));
    port_config_response(w,r,ESD_LINK_STATUS_OK,0xFFu,0u,0u,0u,0u,0u);
}

static bool config_once(uint32_t timeout)
{
    config_work_t w;if(osMessageQueueGet(s_cfg_queue,&w,NULL,timeout)!=osOK)return false;config_response_t r;memset(&r,0,sizeof(r));r.service=w.service;r.message=w.message;r.sequence=w.sequence;
        if(w.service==ESD_LINK_SERVICE_CONTROL&&w.message==ESD_LINK_CONTROL_SET_ENABLE)run_enable(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_SET_ZERO)run_zero(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_PORT_CONFIG_BEGIN)run_port_config_begin(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_PORT_CONFIG_CHUNK)run_port_config_chunk(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_PORT_CONFIG_COMMIT)run_port_config_commit(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_PORT_CONFIG_ABORT)run_port_config_abort(&w,&r);
        else if(w.service==ESD_LINK_SERVICE_CONFIG&&w.message==ESD_LINK_CONFIG_PORT_CONFIG_UPDATE)run_port_config_update(&w,&r);
        else r.status=ESD_LINK_STATUS_UNSUPPORTED_MESSAGE;
        if((w.length>=8u)&&((w.service==ESD_LINK_SERVICE_CONTROL)||(w.service==ESD_LINK_SERVICE_CONFIG)))cache_completion(&w,&r);
        (void)osMessageQueuePut(s_response_queue,&r,0u,osWaitForever);if(s_link_task)xTaskNotifyGive(s_link_task);return true;
}

void TSKPTT_ESDLink_ConfigTask(void *argument)
{
    (void)argument;s_config_task=xTaskGetCurrentTaskHandle();configASSERT(TSKPTT_ESDLink_InitResources());
    for(;;)configASSERT(config_once(osWaitForever));
}

esd_link_result_t TSKPTT_ESDLink_PublishHSKP(uint32_t sequence,uint32_t device_time_us,const uint8_t *report,size_t report_len)
{
    if(!s_ready||!s_connected||s_hskp_period_ms==0u||report_len>472u||(!report&&report_len))return ESD_LINK_ERR_BUSY;
    uint32_t now=now_ms();if((now-s_hskp_last_ms)<s_hskp_period_ms)return ESD_LINK_ERR_BUSY;s_hskp_last_ms=now;
    uint8_t p[ESD_LINK_V1_MAX_PAYLOAD];esd_link_v1_le32_write(&p[0],s_session);esd_link_v1_le32_write(&p[4],device_time_us);copy_bytes(&p[8],report,report_len);
    return esd_link_publish(&s_link,ESD_LINK_SERVICE_HSKP,ESD_LINK_HSKP_REPORT,0u,sequence,p,8u+report_len);
}

esd_link_result_t TSKPTT_ESDLink_SubmitLogRecord(uint64_t timestamp_ms,uint8_t level,const char *source,const char *body)
{
    if(!s_ready||!s_connected||!source||!body)return ESD_LINK_ERR_BUSY;
    size_t sl=bounded_strlen(source,63u),bl=bounded_strlen(body,399u);if(12u+sl+bl>ESD_LINK_V1_MAX_PAYLOAD)bl=ESD_LINK_V1_MAX_PAYLOAD-12u-sl;
    uint8_t p[ESD_LINK_V1_MAX_PAYLOAD];for(size_t i=0;i<8u;++i)p[i]=(uint8_t)(timestamp_ms>>(8u*i));p[8]=level;p[9]=(uint8_t)sl;esd_link_v1_le16_write(&p[10],(uint16_t)bl);copy_bytes(&p[12],(const uint8_t*)source,sl);copy_bytes(&p[12+sl],(const uint8_t*)body,bl);
    return esd_link_submit_log(&s_link,p,12u+sl+bl);
}

uint32_t TSKPTT_ESDLink_CurrentSession(void){return s_session;}
tskptt_esd_control_state_t TSKPTT_ESDLink_ControlState(void){return (tskptt_esd_control_state_t)s_control_state;}
bool TSKPTT_ESDLink_OutputGate(void){return s_output_gate!=0u;}
void TSKPTT_ESDLink_InvalidateSession(void){invalidate_session(TSKPTT_ESD_CONTROL_NO_SESSION);}
void TSKPTT_ESDLink_NotifyDisconnect(void)
{
    (void)__atomic_add_fetch(&s_disconnect_generation,1u,__ATOMIC_RELAXED);
    if(xPortIsInsideInterrupt()==pdTRUE){BaseType_t wake=pdFALSE;if(s_ctrl_task)vTaskNotifyGiveFromISR(s_ctrl_task,&wake);if(s_link_task)vTaskNotifyGiveFromISR(s_link_task,&wake);portYIELD_FROM_ISR(wake);}
    else {if(s_ctrl_task)xTaskNotifyGive(s_ctrl_task);if(s_link_task)xTaskNotifyGive(s_link_task);}
}

#ifdef TSKPTT_ESD_LINK_TESTING
void TSKPTT_ESDLink_TestReset(void)
{
    memset(&s_link,0,sizeof(s_link));memset(&s_cfg,0,sizeof(s_cfg));memset(&s_ports,0,sizeof(s_ports));
    s_cfg_queue=NULL;s_response_queue=NULL;s_link_task=NULL;s_ctrl_task=NULL;s_config_task=NULL;s_control_rate_hz=0u;s_control_period_ms=0u;
    memset(s_transactions,0,sizeof(s_transactions));memset(&s_command,0,sizeof(s_command));memset(&s_cycle,0,sizeof(s_cycle));memset(s_zero,0,sizeof(s_zero));memset(&s_port_staging,0,sizeof(s_port_staging));
    s_session=0u;s_disconnect_generation=0u;s_control_deadline_miss=0u;s_last_command_ms=0u;s_valid_commands=0u;s_rejected_commands=0u;s_last_reject=0u;s_control_state=0u;s_output_gate=0u;s_control_fault_latched=0u;s_seen_deadline_miss=0u;s_event_sequence=0u;s_state_sequence=0u;s_safe_damping_started_ms=0u;s_safe_damping_terminal_state=TSKPTT_ESD_CONTROL_SAFE_DAMPING;memset(s_published_state_sequence,0,sizeof(s_published_state_sequence));s_published_state_count=0u;s_published_state_next=0u;s_last_rx_overflow=0u;s_config_fingerprint=0u;s_hskp_period_ms=0u;s_hskp_last_ms=0u;s_connected=0u;s_configured=0u;s_ready=0u;s_link_disconnect_generation=0u;s_ctrl_disconnect_generation=0u;s_last_status_fault_flags=0u;s_last_status_offline_mask=0u;s_last_status_reject=0u;s_last_status_control_state=0u;s_status_signature_valid=0u;
}
esd_link_result_t TSKPTT_ESDLink_TestStart(TaskHandle_t link_task)
{s_link_task=link_task;if(!TSKPTT_ESDLink_InitResources())return ESD_LINK_ERR_PARAM;return start_link();}
void TSKPTT_ESDLink_TestServiceOnce(void){service_link();}
void TSKPTT_ESDLink_TestControlOnce(void){control_once();}
bool TSKPTT_ESDLink_TestConfigOnce(void){return config_once(0u);}
void TSKPTT_ESDLink_TestPublishState(void){publish_state(now_ms());}
void TSKPTT_ESDLink_TestPublishDevice(void){publish_device(ESD_LINK_QOS_DIAGNOSTIC);}
#endif
