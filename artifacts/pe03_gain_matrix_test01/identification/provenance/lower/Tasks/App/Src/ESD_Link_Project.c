#include "ESD_Link_Project.h"

#include "ESD_Link_BootCounter.h"
#include "ESD_Control_Ports.h"
#include "ESD_Link_ZeroStore.h"
#include "CONF_ESD_Link_Task.h"
#include "bsp_dwt.h"
#include "bsp_host_transport.h"
#include "esd_link_v1_wire.h"
#include "lib_24c02_bl.h"
#include "mem_sections.h"
#include "omx_debug_watch.h"
#include "tskptt_esd_link.h"
#include "tskptt_imu.h"
#include "tskptt_hskp.h"

#include "cmsis_os2.h"
#include "stm32h7xx_hal.h"

#include <string.h>

#define ESD_RX_DRAIN_SIZE 512u
#define ESD_EEPROM_I2C_BUS   0u
#define ESD_EEPROM_ADDR_7BIT 0x50u
#define ESD_IMU_MAX_AGE_MS   50u

OMX_DEBUG_WATCH_BSS volatile omx_dbg_esd_link_t g_omx_dbg_esd_link;
OMX_DEBUG_WATCH_BSS volatile omx_dbg_esd_control_t g_omx_dbg_esd_control;

static OMX_ALIGN32 uint8_t s_tx_storage[
    ESD_LINK_TX_SLOT_COUNT * ESD_LINK_V1_FRAME_SLOT_SIZE];
static OMX_ALIGN32 uint8_t s_rx_storage[ESD_RX_DRAIN_SIZE];
static float s_initial_zero[ESD_PORT_ID_COUNT];
static bool s_zero_storage_healthy;

static uint32_t project_now(void *user) { (void)user; return HAL_GetTick(); }
static uint32_t project_cycles(void *user) { (void)user; return DWT_GetCounter(); }

static void transport_disconnect(void *user)
{ (void)user; TSKPTT_ESDLink_NotifyDisconnect(); }
static bool transport_init(void *user)
{ (void)user; OMX_HostTransport_Init(); OMX_HostTransport_RegisterDisconnectCallback(transport_disconnect,NULL); return true; }
static bool transport_start(void *user, TaskHandle_t task, uint32_t bits)
{
    (void)user; OMX_HostTransport_RegisterTxServiceTask(task);
    return OMX_HostTransport_StartRx(task,bits)==OMX_HOST_TRANSPORT_RX_OK;
}
static size_t transport_read(void *user,uint8_t *out,size_t cap)
{ (void)user; return OMX_HostTransport_Read(out,cap); }
static uint32_t transport_overflow(void *user)
{ (void)user; return OMX_HostTransport_GetRxOverflowBytes(); }
static esd_link_tx_result_t transport_tx(void *user,const uint8_t *data,size_t len)
{
    (void)user;
    switch(OMX_HostTransport_TryTransmit(data,len)) {
    case OMX_HOST_TRANSPORT_WRITE_OK:return ESD_LINK_TX_ACCEPTED;
    case OMX_HOST_TRANSPORT_WRITE_FULL:return ESD_LINK_TX_BUSY;
    case OMX_HOST_TRANSPORT_WRITE_NOT_CONFIGURED:return ESD_LINK_TX_DISCONNECTED;
    default:return ESD_LINK_TX_ERROR;
    }
}
static bool transport_complete(void *user,bool *success)
{ (void)user; return OMX_HostTransport_TakeTxComplete(success); }
static bool transport_connected(void *user)
{ (void)user; return OMX_HostTransport_IsConfigured(); }

static bool imu_snapshot(void *user,uint32_t *time_us,uint8_t *valid_mask,
                         float accel[3],float gyro[3],float quat_xyzw[4])
{
    (void)user;imu_task_snapshot_t snapshot;
    if(!IMU_GetSnapshot(&snapshot)){*valid_mask=0u;memset(accel,0,3u*sizeof(float));memset(gyro,0,3u*sizeof(float));memset(quat_xyzw,0,4u*sizeof(float));return false;}
    const imu_sensor_state_t *source=(snapshot.internal.valid&&snapshot.internal.online)?
        &snapshot.internal:&snapshot.external;
    if(!source->valid||!source->online||g_imu_app_state==IMU_APP_STATE_FAULT||
       (uint32_t)(HAL_GetTick()-source->last_update_ms)>ESD_IMU_MAX_AGE_MS){*valid_mask=0u;memset(accel,0,3u*sizeof(float));memset(gyro,0,3u*sizeof(float));memset(quat_xyzw,0,4u*sizeof(float));return false;}
    *time_us=source->last_update_ms*1000u;*valid_mask=7u;
    memcpy(accel,source->accel,3u*sizeof(float));
    memcpy(gyro,source->gyro,3u*sizeof(float));
    quat_xyzw[0]=source->quat[1];quat_xyzw[1]=source->quat[2];
    quat_xyzw[2]=source->quat[3];quat_xyzw[3]=source->quat[0];
    return true;
}

static bool next_boot_counter(uint32_t *counter_out)
{
    lib_24c02_bl_t eeprom;
    if(lib_24c02_bl_init(&eeprom,(bsp_i2c_bus_id_t)ESD_EEPROM_I2C_BUS,
                         ESD_EEPROM_ADDR_7BIT)!=LIB_24C02_BL_OK)return false;
    uint8_t image[ESD_BOOT_COUNTER_IMAGE_SIZE];
    for(uint8_t slot=0u;slot<ESD_BOOT_COUNTER_SLOT_COUNT;++slot){
        uint8_t *raw=&image[(size_t)slot*ESD_BOOT_COUNTER_SLOT_SIZE];
        if(lib_24c02_bl_read(&eeprom,(uint8_t)(slot*ESD_BOOT_COUNTER_SLOT_SIZE),raw,ESD_BOOT_COUNTER_SLOT_SIZE,20u)!=LIB_24C02_BL_OK)return false;
    }
    uint32_t next=0u;uint8_t slot=0u;
    if(!ESD_Link_BootCounterSelect(image,&next,&slot))return false;
    uint8_t raw[ESD_BOOT_COUNTER_SLOT_SIZE];
    if(!ESD_Link_BootCounterEncode(next,slot,raw))return false;
    if(lib_24c02_bl_write(&eeprom,(uint8_t)(slot*ESD_BOOT_COUNTER_SLOT_SIZE),raw,sizeof(raw),20u,LIB_24C02_BL_TWR_MS)!=LIB_24C02_BL_OK)return false;
    uint8_t verify[ESD_BOOT_COUNTER_SLOT_SIZE];if(lib_24c02_bl_read(&eeprom,(uint8_t)(slot*ESD_BOOT_COUNTER_SLOT_SIZE),verify,sizeof(verify),20u)!=LIB_24C02_BL_OK||memcmp(raw,verify,sizeof(raw))!=0)return false;
    *counter_out=next;return true;
}

static bool next_boot_counter_with_retry(uint32_t *counter_out)
{
    for(uint32_t attempt=0u;attempt<3u;++attempt){
        if(next_boot_counter(counter_out))return true;
        if(attempt+1u<3u)osDelay(2u);
    }
    return false;
}

static esd_zero_store_load_result_t initialize_zero_store_with_retry(
    const esd_port_config_t *ports,size_t count,float *offsets)
{
    esd_zero_store_load_result_t result=ESD_ZERO_STORE_IO_ERROR;
    for(uint32_t attempt=0u;attempt<3u;++attempt){
        result=ESD_Link_ZeroStoreInitialize(ports,count,offsets);
        if(result!=ESD_ZERO_STORE_IO_ERROR)return result;
        if(attempt+1u<3u)osDelay(2u);
    }
    return result;
}

static uint32_t lower_boot_id(uint32_t counter)
{
    uint8_t seed[16];esd_link_v1_le32_write(&seed[0],HAL_GetUIDw0());
    esd_link_v1_le32_write(&seed[4],HAL_GetUIDw1());esd_link_v1_le32_write(&seed[8],HAL_GetUIDw2());
    esd_link_v1_le32_write(&seed[12],counter);uint32_t id=esd_link_v1_crc32(seed,sizeof(seed));return id?id:1u;
}

static bool persist_zero(void *user,const float offsets[ESD_PORT_ID_COUNT],uint32_t mask)
{ (void)user;const bool ok=ESD_Link_ZeroStorePersist(offsets,mask);if(ok)s_zero_storage_healthy=true;return ok; }
static void hskp_config(void *user,bool enabled,uint32_t period_us)
{ (void)user; TSKPTT_HSKP_TelemetryConfigure(enabled,period_us); }

static uint16_t apply_port_config(
    void *user,const esd_port_declaration_t *declarations,size_t count,
    const float current_zero[ESD_PORT_ID_COUNT],
    tskptt_esd_port_config_result_t *result)
{
    (void)user;
    if(result==NULL)return ESD_LINK_STATUS_INTERNAL_ERROR;
    memset(result,0,sizeof(*result));
    esd_control_port_set_t set={0};
    const esd_control_ports_result_t changed=ESD_ControlPorts_Reconfigure(
        declarations,count,current_zero,&set,result->zero_offsets);
    result->ports=set.ports;result->port_count=set.count;
    if(changed==ESD_CONTROL_PORTS_INVALID)return ESD_LINK_STATUS_INVALID_VALUE;
    if(changed==ESD_CONTROL_PORTS_ATTACH_FAILED)return ESD_LINK_STATUS_INTERNAL_ERROR;
    if(changed==ESD_CONTROL_PORTS_ROLLBACK_FAILED){
        result->configuration_lost=true;
        s_zero_storage_healthy=false;result->zero_storage_healthy=false;
        return ESD_LINK_STATUS_INTERNAL_ERROR;
    }
    if(changed==ESD_CONTROL_PORTS_UNCHANGED){
        result->unchanged=true;result->zero_storage_healthy=s_zero_storage_healthy;
        return ESD_LINK_STATUS_OK;
    }

    float stored[ESD_PORT_ID_COUNT];
    const esd_zero_store_load_result_t zero_result=
        initialize_zero_store_with_retry(set.ports,set.count,stored);
    if(zero_result==ESD_ZERO_STORE_LOADED)
        memcpy(result->zero_offsets,stored,sizeof(stored));
    s_zero_storage_healthy=(zero_result==ESD_ZERO_STORE_LOADED)||
        (zero_result==ESD_ZERO_STORE_DEFAULT_EMPTY)||
        (zero_result==ESD_ZERO_STORE_DEFAULT_CONFIG_MISMATCH);
    result->zero_storage_healthy=s_zero_storage_healthy;
    return ESD_LINK_STATUS_OK;
}

static void debug_snapshot(void *user,const esd_link_stats_t *st,
                           tskptt_esd_control_state_t state,uint32_t deadline,
                           uint32_t disconnect,uint32_t link_stack,
                           uint32_t ctrl_stack,uint32_t cfg_stack)
{
    (void)user;omx_dbg_begin(&g_omx_dbg_esd_link.h);
#define COPY(x) g_omx_dbg_esd_link.x=st->x
    COPY(rx_frames);COPY(rx_bytes);COPY(tx_frames);COPY(tx_bytes);COPY(cobs_errors);COPY(crc_errors);
    COPY(length_errors);COPY(version_errors);COPY(flags_errors);COPY(kind_errors);COPY(unknown_service);
    COPY(unknown_message);COPY(rx_overflow);COPY(sequence_gap);COPY(sequence_duplicate);COPY(sequence_out_of_order);
    COPY(request_replay);COPY(request_conflict);COPY(transport_busy);COPY(transport_errors);COPY(transport_disconnects);
    COPY(processing_cycles_total);COPY(processing_cycles_max);COPY(transport_connected);
#undef COPY
    g_omx_dbg_esd_link.started=1u;g_omx_dbg_esd_link.usb_configured=st->transport_connected;
    g_omx_dbg_esd_link.comm_stack_high_water=link_stack;g_omx_dbg_esd_link.control_stack_high_water=ctrl_stack;g_omx_dbg_esd_link.config_stack_high_water=cfg_stack;
    for(size_t i=0u;i<4u;++i){g_omx_dbg_esd_link.qos[i].submitted=st->qos[i].submitted;g_omx_dbg_esd_link.qos[i].sent=st->qos[i].sent;g_omx_dbg_esd_link.qos[i].drop_new=st->qos[i].drop_new;g_omx_dbg_esd_link.qos[i].drop_old=st->qos[i].drop_old;g_omx_dbg_esd_link.qos[i].queue_depth=st->qos[i].queue_depth;g_omx_dbg_esd_link.qos[i].queue_high_water=st->qos[i].queue_high_water;}
    omx_dbg_end(&g_omx_dbg_esd_link.h,HAL_GetTick(),0u);
    omx_dbg_begin(&g_omx_dbg_esd_control.h);g_omx_dbg_esd_control.control_state=state;g_omx_dbg_esd_control.session_id=TSKPTT_ESDLink_CurrentSession();g_omx_dbg_esd_control.disconnect_generation=disconnect;g_omx_dbg_esd_control.deadline_miss=deadline;g_omx_dbg_esd_control.output_gate=TSKPTT_ESDLink_OutputGate()?1u:0u;omx_dbg_end(&g_omx_dbg_esd_control.h,HAL_GetTick(),0u);
}

bool ESD_Link_ProjectConfigure(void)
{
    omx_dbg_zero(&g_omx_dbg_esd_link,sizeof(g_omx_dbg_esd_link));
    omx_dbg_init(&g_omx_dbg_esd_link.h,OMX_DBG_MODULE_ESD_LINK,0u,
                 (uint16_t)sizeof(g_omx_dbg_esd_link));
    omx_dbg_zero(&g_omx_dbg_esd_control,sizeof(g_omx_dbg_esd_control));
    omx_dbg_init(&g_omx_dbg_esd_control.h,OMX_DBG_MODULE_ESD_LINK,1u,
                 (uint16_t)sizeof(g_omx_dbg_esd_control));
    esd_control_port_set_t control_ports={0};
    const bool ports_ok=ESD_ControlPorts_Initialize(&control_ports);
    const esd_zero_store_load_result_t zero_result=ports_ok?
        initialize_zero_store_with_retry(control_ports.ports,control_ports.count,
                                         s_initial_zero):ESD_ZERO_STORE_INVALID;
    const bool zero_store_ok=(zero_result==ESD_ZERO_STORE_LOADED)||
        (zero_result==ESD_ZERO_STORE_DEFAULT_EMPTY)||
        (zero_result==ESD_ZERO_STORE_DEFAULT_CONFIG_MISMATCH);
    s_zero_storage_healthy=zero_store_ok;
    uint32_t counter=0u;const bool boot_ok=next_boot_counter_with_retry(&counter);
    if(!boot_ok)counter=HAL_GetTick()|1u;
    g_omx_dbg_esd_control.boot_counter_init_ok=boot_ok?1u:0u;
    g_omx_dbg_esd_control.zero_store_load_result=(uint32_t)zero_result;
    g_omx_dbg_esd_control.zero_store_init_ok=zero_store_ok?1u:0u;
    g_omx_dbg_esd_control.persistence_init_ok=(boot_ok&&zero_store_ok)?1u:0u;
    const tskptt_esd_link_config_t cfg={
        .transport={.user=NULL,.init=transport_init,.start_rx=transport_start,.rx_read=transport_read,
                    .rx_overflow_bytes=transport_overflow,.tx_submit=transport_tx,
                    .tx_take_complete=transport_complete,.is_connected=transport_connected},
        .now_ms=project_now,.cycles=project_cycles,.clock_user=NULL,
        .tx_storage=s_tx_storage,.tx_storage_size=sizeof(s_tx_storage),
        .rx_storage=s_rx_storage,.rx_storage_size=sizeof(s_rx_storage),
        .ports=ports_ok?control_ports.ports:NULL,
        .port_count=ports_ok?control_ports.count:0u,
        .control_rate_hz=OMX_ESD_CONTROL_RATE_HZ,
        .initial_zero_offsets=s_initial_zero,
        .lower_boot_id=lower_boot_id(counter),
        .boot_storage_healthy=boot_ok,
        .zero_storage_healthy=ports_ok&&zero_store_ok,
        .configuration_healthy=ports_ok,.layout_id=1u,.schema_id=1u,
        .device_capabilities=0x0000001Fu,.imu_snapshot=imu_snapshot,
        .persist_zero=persist_zero,.hskp_config=hskp_config,
        .snapshot=debug_snapshot,.apply_port_config=apply_port_config,
        .project_user=NULL};
    return TSKPTT_ESDLink_Configure(&cfg)==ESD_LINK_OK;
}
