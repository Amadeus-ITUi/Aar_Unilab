#include "esd_link_bridge/time_utils.hpp"

#include <gtest/gtest.h>

namespace {

TEST(TimeUtils, ExpiresOnlyWhenTimestampIsActuallyOld) {
  constexpr std::uint64_t timeout = 40'000'000;
  EXPECT_FALSE(esd_link_bridge::timestamp_is_stale(100, 101, timeout));
  EXPECT_FALSE(esd_link_bridge::timestamp_is_stale(timeout, 1, timeout));
  EXPECT_TRUE(esd_link_bridge::timestamp_is_stale(timeout + 2, 1, timeout));
}

TEST(TimeUtils, MissingTimestampIsStale) {
  EXPECT_TRUE(esd_link_bridge::timestamp_is_stale(100, 0, 40));
}

TEST(TimeUtils, SaturatesFutureTimestampAgeAtZero) {
  EXPECT_EQ(esd_link_bridge::elapsed_ns_saturated(100, 101), 0U);
  EXPECT_EQ(esd_link_bridge::elapsed_ns_saturated(101, 100), 1U);
}

}  // namespace
