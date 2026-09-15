#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace esd_link_bridge {

// One producer, one consumer, latest-value-only triple buffer. Publishing never
// waits for the consumer; the consumer either sees a complete new snapshot or
// no update. Intermediate generations may intentionally be overwritten.
template <typename T>
class LatestMailbox {
 public:
  void publish(const T &value) {
    slots_[back_].value = value;
    slots_[back_].generation = ++producer_generation_;
    back_ = middle_.exchange(back_, std::memory_order_acq_rel);
    dirty_.store(true, std::memory_order_release);
  }

  bool consume(T &value) {
    if (!dirty_.exchange(false, std::memory_order_acq_rel)) return false;
    front_ = middle_.exchange(front_, std::memory_order_acq_rel);
    if (slots_[front_].generation <= consumer_generation_) return false;
    consumer_generation_ = slots_[front_].generation;
    value = slots_[front_].value;
    return true;
  }

 private:
  struct Slot {
    T value{};
    std::uint64_t generation{0};
  };
  std::array<Slot, 3> slots_{};
  std::size_t front_{0};
  std::atomic<std::size_t> middle_{1};
  std::size_t back_{2};
  std::atomic<bool> dirty_{false};
  std::uint64_t producer_generation_{0};
  std::uint64_t consumer_generation_{0};
};

}  // namespace esd_link_bridge
