#pragma once

#include <cstdint>

namespace esd_link_bridge {

// Timestamps are written and read by different threads. The writer may sample
// the monotonic clock after the reader sampled `now`, so unsigned subtraction
// must not interpret a slightly newer timestamp as an enormous elapsed time.
constexpr std::uint64_t elapsed_ns_saturated(
    std::uint64_t now_ns, std::uint64_t timestamp_ns) {
  return now_ns >= timestamp_ns ? now_ns - timestamp_ns : 0;
}

constexpr bool timestamp_is_stale(
    std::uint64_t now_ns, std::uint64_t timestamp_ns,
    std::uint64_t timeout_ns) {
  return timestamp_ns == 0 ||
         (now_ns >= timestamp_ns && now_ns - timestamp_ns > timeout_ns);
}

}  // namespace esd_link_bridge
