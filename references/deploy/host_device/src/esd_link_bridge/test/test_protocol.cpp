#include "esd_link_bridge/protocol.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

using namespace esd_link_bridge::protocol;

namespace {
std::vector<std::uint8_t> from_hex(const std::string &hex) {
  std::vector<std::uint8_t> result;
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    result.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  }
  return result;
}
}

TEST(Protocol, OpenSessionMatchesFrozenPythonCodec) {
  Frame frame;
  ASSERT_TRUE(build_open_session(0x12345678U, 1, 1, 0, frame));
  frame.sequence = 0x10203040U;
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(frame, wire));
  const auto expected = from_hex(
      "0003a50101040101110b4030201078563412010102010101010101010101058305c3cc00");
  EXPECT_EQ(std::vector<std::uint8_t>(wire.bytes.begin(), wire.bytes.begin() + wire.size), expected);
}

TEST(Protocol, EnableMatchesFrozenPythonCodec) {
  Frame frame;
  ASSERT_TRUE(build_set_enable(0x55667788U, 0x99AABBCCU, true, frame));
  frame.sequence = 0x11223344U;
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(frame, wire));
  const auto expected = from_hex(
      "0003a50101040202121b4433221188776655ccbbaa990108010203040506070802edd6c400");
  EXPECT_EQ(std::vector<std::uint8_t>(wire.bytes.begin(), wire.bytes.begin() + wire.size), expected);
}

TEST(Protocol, SetZeroMatchesFrozenPythonCodec) {
  const std::array<std::uint8_t, 2> ports{1, 6};
  const std::array<float, 2> positions{1.25F, -2.5F};
  Frame frame;
  ASSERT_TRUE(build_set_zero(0x11223344U, 0x55667788U, true,
                             ports.data(), positions.data(), ports.size(), frame));
  frame.sequence = 0x10203040U;
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(frame, wire));
  const auto expected = from_hex(
      "0003a5010104040114104030201044332211887766550102010104a03f06010720c00ad7aca100");
  EXPECT_EQ(std::vector<std::uint8_t>(wire.bytes.begin(), wire.bytes.begin() + wire.size), expected);
}

TEST(Protocol, FullActuatorCommandMatchesFrozenPythonCodec) {
  std::array<ActuatorCommand, 8> commands{};
  for (std::size_t i = 0; i < commands.size(); ++i) {
    const float id = static_cast<float>(i + 1);
    commands[i] = {static_cast<std::uint8_t>(i + 1), id, -id / 10.0F,
                   2.0F + id, 0.1F * id, 0.0F};
  }
  Frame frame;
  ASSERT_TRUE(build_actuator_command(0x11223344U, 1, 0x55667788U,
                                     0x99AABBCCU, commands, frame));
  frame.sequence = 0x10203040U;
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(frame, wire));
  const auto expected = from_hex(
      "0004a50102040201b70a4030201044332211010b88776655ccbbaa9908010107803fcdccccbd01074040cdcccc3d010101020201010640cdcc4cbe01078040cdcc4c3e0101010203010740409a9999be0107a0409a99993e010101020401078040cdccccbe0107c040cdcccc3e01010102050103a040010102bf0103e0400101023f01010102060107c0409a9919bf010106419a99193f01010102070107e040333333bf010710413333333f010101020801010641cdcc4cbf01072041cdcc4c3f0101010576ea3fca00");
  EXPECT_EQ(std::vector<std::uint8_t>(wire.bytes.begin(), wire.bytes.begin() + wire.size), expected);
}

TEST(Protocol, StreamingHandlesEveryFragmentBoundary) {
  Frame source;
  ASSERT_TRUE(build_open_session(0x12345678U, 1, 1, 0, source));
  source.sequence = 9;
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(source, wire));
  for (std::size_t split = 0; split <= wire.size; ++split) {
    int called = 0;
    StreamDecoder decoder([&](const Frame &frame, std::uint64_t stamp) {
      ++called;
      EXPECT_EQ(frame.sequence, 9U);
      EXPECT_EQ(stamp, split == wire.size ? 77U : 88U);
    });
    decoder.feed(wire.bytes.data(), split, 77);
    decoder.feed(wire.bytes.data() + split, wire.size - split, 88);
    EXPECT_EQ(called, 1) << "split=" << split;
    EXPECT_EQ(decoder.frames(), 1U);
  }
}

TEST(Protocol, BadCrcIsCountedAndStreamRecovers) {
  Frame source;
  ASSERT_TRUE(build_open_session(0x12345678U, 1, 1, 0, source));
  WireBuffer wire;
  ASSERT_TRUE(encode_frame(source, wire));
  auto bad = wire;
  bad.bytes[bad.size / 2] ^= 0x40;
  int called = 0;
  StreamDecoder decoder([&](const Frame &, std::uint64_t) { ++called; });
  decoder.feed(bad.bytes.data(), bad.size, 1);
  decoder.feed(wire.bytes.data(), wire.size, 2);
  EXPECT_EQ(called, 1);
  EXPECT_EQ(decoder.frames(), 1U);
  EXPECT_EQ(decoder.crc_errors() + decoder.cobs_errors(), 1U);
}

TEST(Protocol, NoiseOverlongAndMergedFramesResynchronize) {
  Frame first;
  Frame second;
  ASSERT_TRUE(build_open_session(1, 1, 1, 0, first));
  ASSERT_TRUE(build_open_session(2, 1, 1, 0, second));
  first.sequence = 0xFFFFFFFFU;
  second.sequence = 0;
  WireBuffer one;
  WireBuffer two;
  ASSERT_TRUE(encode_frame(first, one));
  ASSERT_TRUE(encode_frame(second, two));
  std::vector<std::uint8_t> stream{0x55, 0xAA, 0x11, 0};
  stream.insert(stream.end(), kMaxEncoded + 10, 0x33);
  stream.push_back(0);
  stream.insert(stream.end(), one.bytes.begin(), one.bytes.begin() + one.size);
  stream.insert(stream.end(), two.bytes.begin(), two.bytes.begin() + two.size);
  std::vector<std::uint32_t> sequences;
  StreamDecoder decoder([&](const Frame &frame, std::uint64_t) {
    sequences.push_back(frame.sequence);
  });
  for (std::size_t i = 0; i < stream.size();) {
    const std::size_t chunk = std::min<std::size_t>((i * 17U) % 23U + 1U, stream.size() - i);
    decoder.feed(stream.data() + i, chunk, i);
    i += chunk;
  }
  EXPECT_EQ(sequences, (std::vector<std::uint32_t>{0xFFFFFFFFU, 0U}));
  EXPECT_GE(decoder.cobs_errors(), 1U);
}

TEST(Protocol, ActuatorCommandRequiresAllEightUniquePorts) {
  std::array<ActuatorCommand, 8> commands{};
  for (std::size_t i = 0; i < commands.size(); ++i) {
    commands[i] = {static_cast<std::uint8_t>(i + 1), static_cast<float>(i), 0, 1, 0.1F, 0};
  }
  Frame frame;
  EXPECT_TRUE(build_actuator_command(1, 1, 42, 0x1234, commands, frame));
  commands[7].port_id = 7;
  EXPECT_FALSE(build_actuator_command(1, 1, 42, 0x1234, commands, frame));
}
