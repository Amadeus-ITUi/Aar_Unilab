#include "esd_link_bridge/protocol.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

namespace {
using esd_link_bridge::protocol::Frame;
using esd_link_bridge::protocol::StreamDecoder;
using esd_link_bridge::protocol::WireBuffer;
using Clock = std::chrono::steady_clock;

constexpr const char *kDefaultDevice =
    "/dev/serial/by-id/usb-Sirin_Systems___OmniX_Robotics_ESD-SLAVE_314137523333-if00";

bool write_frame(int fd, Frame frame, std::uint32_t sequence) {
  frame.sequence = sequence;
  WireBuffer wire;
  if (!esd_link_bridge::protocol::encode_frame(frame, wire)) return false;
  std::size_t offset = 0;
  while (offset < wire.size) {
    const auto written = ::write(fd, wire.bytes.data() + offset, wire.size - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && (errno == EINTR || errno == EAGAIN)) {
      pollfd item{fd, POLLOUT, 0};
      if (::poll(&item, 1, 100) >= 0) continue;
    }
    return false;
  }
  return true;
}

int open_device(const char *path) {
  const int fd = ::open(path, O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return -1;
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ::close(fd);
    return -1;
  }
  termios tty{};
  if (::tcgetattr(fd, &tty) != 0) {
    ::close(fd);
    return -1;
  }
  ::cfmakeraw(&tty);
  ::cfsetispeed(&tty, B115200);
  ::cfsetospeed(&tty, B115200);
  tty.c_cflag |= CLOCAL | CREAD;
  tty.c_cflag &= ~CRTSCTS;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;
  if (::tcsetattr(fd, TCSANOW, &tty) != 0) {
    ::close(fd);
    return -1;
  }
  ::tcflush(fd, TCIOFLUSH);
  return fd;
}
}  // namespace

int main(int argc, char **argv) {
  const char *device = argc > 1 ? argv[1] : kDefaultDevice;
  const int fd = open_device(device);
  if (fd < 0) {
    std::fprintf(stderr, "cannot exclusively open %s: %s\n", device, std::strerror(errno));
    return 2;
  }

  const auto nonce = [] {
    std::random_device source;
    auto value = static_cast<std::uint32_t>(source());
    return value == 0 ? 1U : value;
  }();
  std::uint32_t session_id = 0;
  bool disable_done = false;
  std::uint16_t disable_status = 0xFFFF;
  constexpr std::uint32_t kTransaction = 1;
  StreamDecoder decoder([&](const Frame &frame, std::uint64_t) {
    esd_link_bridge::protocol::SessionInfo session;
    if (esd_link_bridge::protocol::parse_session_info(frame, session) &&
        session.status == 0 && session.host_nonce == nonce) {
      session_id = session.session_id;
      return;
    }
    esd_link_bridge::protocol::TransactionResult result;
    if (esd_link_bridge::protocol::parse_transaction_result(frame, result) &&
        result.transaction_id == kTransaction) {
      disable_status = result.status;
      disable_done = true;
    }
  });

  Frame open;
  if (!esd_link_bridge::protocol::build_open_session(nonce, 1, 1, 0, open) ||
      !write_frame(fd, open, 1)) {
    std::fprintf(stderr, "failed to send OPEN_SESSION\n");
    ::close(fd);
    return 3;
  }

  const auto deadline = Clock::now() + std::chrono::seconds(3);
  std::array<std::uint8_t, 2048> bytes{};
  bool disable_sent = false;
  while (Clock::now() < deadline && !disable_done) {
    pollfd item{fd, POLLIN | POLLERR | POLLHUP, 0};
    const int ready = ::poll(&item, 1, 100);
    if (ready < 0 && errno != EINTR) break;
    if (item.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
    if (item.revents & POLLIN) {
      const auto count = ::read(fd, bytes.data(), bytes.size());
      if (count > 0) decoder.feed(bytes.data(), static_cast<std::size_t>(count), 0);
    }
    if (session_id != 0 && !disable_sent) {
      Frame disable;
      if (!esd_link_bridge::protocol::build_set_enable(session_id, kTransaction, false, disable) ||
          !write_frame(fd, disable, 2)) break;
      disable_sent = true;
    }
  }
  ::flock(fd, LOCK_UN);
  ::close(fd);
  if (!disable_done || disable_status != 0) {
    std::fprintf(stderr, "ESD-Link disable was not confirmed (status=%u)\n", disable_status);
    return 4;
  }
  std::printf("ESD-Link confirmed P1-P8 DISABLED\n");
  return 0;
}
