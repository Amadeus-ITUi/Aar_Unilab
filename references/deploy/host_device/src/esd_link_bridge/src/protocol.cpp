#include "esd_link_bridge/protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace esd_link_bridge::protocol {
namespace {

std::uint16_t le16(const std::uint8_t *p) {
  return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
}
std::uint32_t le32(const std::uint8_t *p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
void put16(std::uint8_t *p, std::uint16_t v) {
  p[0] = static_cast<std::uint8_t>(v); p[1] = static_cast<std::uint8_t>(v >> 8);
}
void put32(std::uint8_t *p, std::uint32_t v) {
  p[0] = static_cast<std::uint8_t>(v); p[1] = static_cast<std::uint8_t>(v >> 8);
  p[2] = static_cast<std::uint8_t>(v >> 16); p[3] = static_cast<std::uint8_t>(v >> 24);
}
float get_float(const std::uint8_t *p) {
  const auto bits = le32(p); float value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
void put_float(std::uint8_t *p, float value) {
  std::uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); put32(p, bits);
}

bool append_u8(Frame &f, std::uint8_t value) {
  if (f.payload_len >= f.payload.size()) return false;
  f.payload[f.payload_len++] = value; return true;
}
bool append_u16(Frame &f, std::uint16_t value) {
  if (f.payload_len + 2 > f.payload.size()) return false;
  put16(&f.payload[f.payload_len], value); f.payload_len += 2; return true;
}
bool append_u32(Frame &f, std::uint32_t value) {
  if (f.payload_len + 4 > f.payload.size()) return false;
  put32(&f.payload[f.payload_len], value); f.payload_len += 4; return true;
}
bool append_float(Frame &f, float value) {
  if (!std::isfinite(value) || f.payload_len + 4 > f.payload.size()) return false;
  put_float(&f.payload[f.payload_len], value); f.payload_len += 4; return true;
}

DecodeError decode_raw(const std::uint8_t *raw, std::size_t size, Frame &out) {
  if (size < 16 || size > kMaxRaw) return DecodeError::RawLength;
  if (raw[0] != kMagic) return DecodeError::Magic;
  if (raw[1] != kWireVersion) return DecodeError::Version;
  if (raw[2] > static_cast<std::uint8_t>(Kind::Event)) return DecodeError::Kind;
  if ((raw[3] & ~0x01U) != 0) return DecodeError::Flags;
  const auto payload_len = le16(raw + 6);
  if (payload_len > kMaxPayload || size != 12U + payload_len + 4U) return DecodeError::PayloadLength;
  if (le32(raw + 12 + payload_len) != crc32(raw, 12 + payload_len)) return DecodeError::Crc;
  out.kind = static_cast<Kind>(raw[2]); out.flags = raw[3]; out.service = raw[4];
  out.message = raw[5]; out.sequence = le32(raw + 8); out.payload_len = payload_len;
  std::copy_n(raw + 12, payload_len, out.payload.begin());
  return DecodeError::None;
}

}  // namespace

std::uint32_t crc32(const std::uint8_t *data, std::size_t size) {
  static constexpr std::uint32_t table[16] = {
      0x00000000U,0x1DB71064U,0x3B6E20C8U,0x26D930ACU,0x76DC4190U,0x6B6B51F4U,
      0x4DB26158U,0x5005713CU,0xEDB88320U,0xF00F9344U,0xD6D6A3E8U,0xCB61B38CU,
      0x9B64C2B0U,0x86D3D2D4U,0xA00AE278U,0xBDBDF21CU};
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i]; crc = (crc >> 4) ^ table[crc & 0x0F]; crc = (crc >> 4) ^ table[crc & 0x0F];
  }
  return crc ^ 0xFFFFFFFFU;
}

DecodeError decode_encoded(const std::uint8_t *encoded, std::size_t size, Frame &out) {
  if (size == 0) return DecodeError::Cobs;
  std::array<std::uint8_t, kMaxRaw> raw{}; std::size_t r = 0; std::size_t i = 0;
  while (i < size) {
    const auto code = encoded[i++]; if (code == 0) return DecodeError::Cobs;
    const auto count = static_cast<std::size_t>(code - 1);
    if (i + count > size || r + count > raw.size()) return DecodeError::Cobs;
    std::copy_n(encoded + i, count, raw.begin() + r); i += count; r += count;
    if (code != 0xFF && i < size) { if (r >= raw.size()) return DecodeError::Overflow; raw[r++] = 0; }
  }
  return decode_raw(raw.data(), r, out);
}

StreamDecoder::StreamDecoder(Callback callback) : callback_(std::move(callback)) {}
void StreamDecoder::reset() { encoded_len_ = 0; in_frame_ = false; dropping_ = false; }
void StreamDecoder::feed(const std::uint8_t *data, std::size_t size, std::uint64_t receive_time_ns) {
  for (std::size_t i = 0; i < size; ++i) {
    const auto value = data[i];
    if (value == 0) {
      if (in_frame_ && encoded_len_ > 0 && !dropping_) finish(receive_time_ns);
      in_frame_ = true; dropping_ = false; encoded_len_ = 0; continue;
    }
    if (!in_frame_ || dropping_) continue;
    if (encoded_len_ == encoded_.size()) {
      dropping_ = true; ++cobs_errors_; last_error_ = DecodeError::Overflow;
    } else encoded_[encoded_len_++] = value;
  }
}
void StreamDecoder::finish(std::uint64_t receive_time_ns) {
  Frame frame; const auto error = decode_encoded(encoded_.data(), encoded_len_, frame); last_error_ = error;
  if (error == DecodeError::None) { ++frames_; callback_(frame, receive_time_ns); }
  else if (error == DecodeError::Crc) ++crc_errors_;
  else ++cobs_errors_;
}

bool encode_frame(const Frame &frame, WireBuffer &out) {
  if (frame.payload_len > kMaxPayload) return false;
  std::array<std::uint8_t, kMaxRaw> raw{};
  raw[0]=kMagic; raw[1]=kWireVersion; raw[2]=static_cast<std::uint8_t>(frame.kind);
  raw[3]=frame.flags; raw[4]=frame.service; raw[5]=frame.message;
  put16(raw.data()+6, static_cast<std::uint16_t>(frame.payload_len)); put32(raw.data()+8, frame.sequence);
  std::copy_n(frame.payload.begin(), frame.payload_len, raw.begin()+12);
  const auto raw_len=12+frame.payload_len+4; put32(raw.data()+12+frame.payload_len, crc32(raw.data(),12+frame.payload_len));
  std::size_t write=0; out.bytes[write++]=0; std::size_t code_index=write++; std::uint8_t code=1;
  for(std::size_t i=0;i<raw_len;++i){ const auto b=raw[i]; if(b==0){out.bytes[code_index]=code;code_index=write++;code=1;}
    else {out.bytes[write++]=b;if(++code==0xFF){out.bytes[code_index]=code;code_index=write++;code=1;}}
    if(write>=out.bytes.size()-1)return false; }
  out.bytes[code_index]=code; out.bytes[write++]=0; out.size=write; return true;
}

bool build_open_session(std::uint32_t nonce, std::uint16_t layout_id, std::uint16_t schema_id,
                        std::uint32_t expected_fingerprint, Frame &out) {
  if (!nonce) return false;
  out={}; out.kind=Kind::Request; out.service=kServiceSession; out.message=kSessionOpen;
  return append_u32(out,nonce)&&append_u8(out,1)&&append_u16(out,layout_id)&&append_u16(out,schema_id)&&
         append_u32(out,expected_fingerprint)&&append_u32(out,0);
}
bool build_set_enable(std::uint32_t session_id, std::uint32_t transaction_id, bool enable, Frame &out) {
  out={}; out.kind=Kind::Request; out.service=kServiceControl; out.message=kSetEnable;
  if(!append_u32(out,session_id)||!append_u32(out,transaction_id)||!append_u8(out,enable?1:0))return false;
  if(enable){if(!append_u8(out,8))return false;for(std::uint8_t p=1;p<=8;++p)if(!append_u8(out,p))return false;}
  else return append_u8(out,1)&&append_u8(out,0xFF);
  return true;
}
bool build_set_zero(std::uint32_t session_id,std::uint32_t transaction_id,bool persist,
                    const std::uint8_t *ports,const float *assigned,std::size_t count,Frame &out){
  if(!ports||!assigned||count==0||count>8)return false;
  out={};out.kind=Kind::Request;out.service=kServiceConfig;out.message=kSetZero;
  if(!append_u32(out,session_id)||!append_u32(out,transaction_id)||!append_u8(out,persist?1:0)||!append_u8(out,count))return false;
  std::uint32_t seen=0;for(std::size_t i=0;i<count;++i){if(ports[i]<1||ports[i]>8||(seen&(1U<<ports[i]))||!std::isfinite(assigned[i]))return false;seen|=1U<<ports[i];if(!append_u8(out,ports[i])||!append_float(out,assigned[i]))return false;}return true;
}
bool build_actuator_command(std::uint32_t session_id,std::uint16_t layout_id,std::uint32_t source,
                            std::uint32_t fingerprint,const std::array<ActuatorCommand,8>& commands,Frame& out){
  out={};out.kind=Kind::Event;out.service=kServiceControl;out.message=kActuatorCommand;
  if(!append_u32(out,session_id)||!append_u16(out,layout_id)||!append_u32(out,source)||!append_u32(out,fingerprint)||!append_u8(out,8))return false;
  std::uint32_t seen=0;for(const auto& c:commands){if(c.port_id<1||c.port_id>8||(seen&(1U<<c.port_id))||c.kp<0||c.kd<0)return false;seen|=1U<<c.port_id;
    if(!append_u8(out,c.port_id)||!append_float(out,c.position)||!append_float(out,c.velocity)||!append_float(out,c.kp)||!append_float(out,c.kd)||!append_float(out,c.effort))return false;}return seen==0x1FEU;
}

std::uint16_t response_status(const Frame &f){return f.payload_len>=2?le16(f.payload.data()):0xFFFF;}
bool parse_session_info(const Frame&f,SessionInfo&o){if(f.kind!=Kind::Response||f.service!=kServiceSession||f.message!=kSessionOpen||f.payload_len!=34)return false;const auto*p=f.payload.data();o.status=le16(p);p+=2;o.host_nonce=le32(p);p+=4;o.lower_boot_id=le32(p);p+=4;o.session_id=le32(p);p+=4;o.business_version=*p++;o.layout_id=le16(p);p+=2;o.active_port_mask=le32(p);p+=4;o.schema_id=le16(p);p+=2;o.config_fingerprint=le32(p);p+=4;o.control_rate_hz=le16(p);p+=2;o.device_capabilities=le32(p);p+=4;o.control_state=*p;return true;}
bool parse_device_status(const Frame&f,DeviceStatus&o){if(f.kind!=Kind::Event||f.service!=kServiceState||f.message!=kDeviceStatus||f.payload_len!=37)return false;const auto*p=f.payload.data();o.session_id=le32(p);p+=4;o.lower_boot_id=le32(p);p+=4;o.control_state=*p++;o.fault_flags=le32(p);p+=4;o.last_reject_code=le16(p);p+=2;o.valid_command_count=le32(p);p+=4;o.invalid_frame_count=le32(p);p+=4;o.rejected_command_count=le32(p);p+=4;o.command_age_ms=le16(p);p+=2;o.active_port_mask=le32(p);p+=4;o.offline_port_mask=le32(p);return true;}
bool parse_transaction_result(const Frame&f,TransactionResult&o){if(f.kind!=Kind::Response||f.payload_len!=37)return false;const auto*p=f.payload.data();o.status=le16(p);p+=2;o.requested_session_id=le32(p);p+=4;o.current_session_id=le32(p);p+=4;o.transaction_id=le32(p);p+=4;o.failed_port_id=*p++;o.detail_code=le16(p);p+=2;o.config_fingerprint=le32(p);p+=4;o.requested_port_mask=le32(p);p+=4;o.affected_port_mask=le32(p);p+=4;o.verified_port_mask=le32(p);p+=4;o.rollback_port_mask=le32(p);return true;}
bool parse_robot_state(const Frame&f,RobotState&o){if(f.kind!=Kind::Event||f.service!=kServiceState||f.message!=kRobotState||f.payload_len<21)return false;const auto*p=f.payload.data();const auto*end=p+f.payload_len;o={};o.session_id=le32(p);p+=4;o.schema_id=le16(p);p+=2;o.sample_time_us=le32(p);p+=4;o.state_sample_seq=le32(p);p+=4;o.last_applied_command_seq=le32(p);p+=4;o.command_status_flags=*p++;o.control_state=*p++;const auto blocks=*p++;if(blocks>16)return false;
  std::uint32_t seen=0,port_mask=0;std::array<float,9> efforts{};std::array<bool,9> have_effort{};
  for(std::uint8_t b=0;b<blocks;++b){if(p+3>end)return false;const auto type=*p++;const auto len=le16(p);p+=2;if(type>31||(seen&(1U<<type))||p+len>end)return false;seen|=1U<<type;const auto*block_end=p+len;
    if(type==1){if(len!=29&&len!=41)return false;o.imu_valid_mask=*p++;for(auto&v:o.gyro){v=get_float(p);p+=4;}for(auto&v:o.quaternion){v=get_float(p);p+=4;}if(len==41)for(auto&v:o.accel){v=get_float(p);p+=4;}}
    else if(type==2){if(p>=block_end)return false;const auto count=*p++;if(count>8||len!=1+10U*count)return false;o.port_count=count;for(std::size_t i=0;i<count;++i){auto&v=o.ports[i];v.id=*p++;if(v.id<1||v.id>8||(port_mask&(1U<<v.id)))return false;port_mask|=1U<<v.id;v.valid_mask=*p++;v.position=get_float(p);p+=4;v.velocity=get_float(p);p+=4;}}
    else if(type==3){if(p>=block_end)return false;const auto count=*p++;if(count>8||len!=1+5U*count)return false;for(std::size_t i=0;i<count;++i){const auto id=*p++;if(id<1||id>8||have_effort[id])return false;efforts[id]=get_float(p);p+=4;have_effort[id]=true;}}
    p=block_end; }
  if(p!=end||o.port_count!=8||port_mask!=0x1FEU)return false;
  for(auto&v:o.ports)if(have_effort[v.id])v.effort=efforts[v.id];
  return true;}

const char*status_name(std::uint16_t s){switch(s){case 0:return"OK";case 1:return"INVALID_LENGTH";case 2:return"UNSUPPORTED_VERSION";case 3:return"UNSUPPORTED_SERVICE";case 4:return"UNSUPPORTED_MESSAGE";case 5:return"SESSION_MISMATCH";case 6:return"INVALID_PORT_ID";case 7:return"DUPLICATE_PORT_ID";case 8:return"INVALID_VALUE";case 9:return"INVALID_STATE";case 10:return"LAYOUT_MISMATCH";case 11:return"SCHEMA_MISMATCH";case 12:return"CONFIG_MISMATCH";case 13:return"BUSY";case 14:return"PERSIST_FAILED";case 15:return"INTERNAL_ERROR";case 16:return"DEVICE_OFFLINE";case 17:return"VERIFY_FAILED";case 20:return"CONFLICT";case 21:return"UNSUPPORTED_CAPABILITY";default:return"UNKNOWN";}}

}  // namespace esd_link_bridge::protocol
