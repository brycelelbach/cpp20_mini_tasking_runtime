#include <jthread>
#include <atomic>
#include <atomic_wait>
#include <mutex>
#include <iostream>

struct thread_group {
private:
  std::vector<std::jthread> members;

public:
  thread_group(thread_group const&) = delete;
  thread_group& operator=(thread_group const&) = delete;

  template <typename Invocable>
  thread_group(std::size_t count, Invocable f) {
    // TODO: Something something ranges, something something no raw loops.
    members.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      members.emplace_back(std::jthread(f));
    }
  }
};

struct ticket_mutex {
private:
  alignas(64) std::atomic<int> in  = ATOMIC_VAR_INIT(0);
  alignas(64) std::atomic<int> out = ATOMIC_VAR_INIT(0);

public:
  void lock() noexcept {
    auto const my = in.fetch_add(1, std::memory_order_acquire);
    while (true) {
      auto const now = out.load(std::memory_order_acquire);
      if (now == my) return;
      atomic_wait_explicit(&out, now, std::memory_order_relaxed);
    }
  }

  void unlock() noexcept {
    out.fetch_add(1, std::memory_order_release);
    atomic_notify_all(&out);
  }
};

int main() {
  ticket_mutex mtx;
  std::size_t count(0);
 
  { 
    thread_group tg(8, 
      [&] {
        std::scoped_lock l(mtx);
        ++count;
      }
    );
  }

  std::cout << count << std::endl;
}

