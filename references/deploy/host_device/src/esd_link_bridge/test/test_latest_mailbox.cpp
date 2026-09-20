#include "esd_link_bridge/latest_mailbox.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace {
struct Snapshot {
  std::uint64_t generation{0};
  std::uint64_t inverse{~0ULL};
  std::uint64_t checksum{0};
};
}  // namespace

TEST(LatestMailbox, ConsumerOnlyObservesCompleteSnapshotsAndFinalGeneration) {
  esd_link_bridge::LatestMailbox<Snapshot> mailbox;
  constexpr std::uint64_t kLast = 200000;
  std::atomic<bool> finished{false};
  std::atomic<bool> torn{false};
  std::uint64_t consumed = 0;
  std::thread consumer([&] {
    Snapshot value;
    while (!finished.load(std::memory_order_acquire) || consumed != kLast) {
      if (!mailbox.consume(value)) {
        std::this_thread::yield();
        continue;
      }
      if (value.inverse != ~value.generation ||
          value.checksum != value.generation * 0x9E3779B185EBCA87ULL) {
        torn.store(true, std::memory_order_release);
      }
      EXPECT_GE(value.generation, consumed);
      consumed = value.generation;
    }
  });
  for (std::uint64_t generation = 1; generation <= kLast; ++generation) {
    mailbox.publish({generation, ~generation,
                     generation * 0x9E3779B185EBCA87ULL});
  }
  finished.store(true, std::memory_order_release);
  consumer.join();
  EXPECT_FALSE(torn.load());
  EXPECT_EQ(consumed, kLast);
}
