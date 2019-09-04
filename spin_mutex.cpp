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

struct spin_mutex {
private:
	std::atomic<bool> flag = ATOMIC_VAR_INIT(0);

public:
	void lock() noexcept {
		while (flag.exchange(true, std::memory_order_acquire))
			atomic_wait_explicit(&flag, true, std::memory_order_relaxed);
	}

	void unlock() noexcept {
		flag.store(0, std::memory_order_release);
		atomic_notify_one(&flag);
	}
};

int main() {
  spin_mutex mtx;
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

