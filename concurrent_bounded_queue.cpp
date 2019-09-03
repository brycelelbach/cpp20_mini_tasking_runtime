#include <queue>
#include <mutex>
#include <jthread>
#include <stop_token>
#include <atomic_wait>
#include <latch>
#include <iostream>
#include <semaphore>
#include <functional>
#include <optional>

#include <cjdb/concepts.hpp>

#if !defined(NDEBUG) && !defined(__NO_LOGGING)
  #define LOG(...) std::cout << __VA_ARGS__ "\n"
#else
  #define LOG(...)
#endif

namespace std { using namespace cjdb; }

struct thread_group {
private:
  std::vector<std::jthread> members;

public:
  thread_group(thread_group const&) = delete;
  thread_group& operator=(thread_group const&) = delete;

  thread_group(std::size_t count, std::invocable auto f) {
    // TODO: Something something ranges, something something no raw loops.
    members.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      members.emplace_back(std::jthread(f));
    }
  }

  thread_group(std::size_t count, std::invocable<std::stop_token> auto f) {
    // TODO: Something something ranges, something something no raw loops.
    members.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      members.emplace_back(std::jthread(f));
    }
  }
};

// TODO: Need a way to shut it down.
// TODO: For forward progress, need to try_acquire and then pop some work from
// the queue when queuing.
// TODO: Should N be ptrdiff_t or size_t?

template <typename T, std::size_t N>
struct concurrent_bounded_queue {
private:
  std::queue<T> items; // TODO: Would prefer a fixed-sized queue.
  std::mutex items_mtx;
  std::counting_semaphore<N> items_produced  { 0 };
  std::counting_semaphore<N> remaining_space { N };

public:
  constexpr concurrent_bounded_queue() = default;

  // Enqueue one entry.
  template <typename U>
  void enqueue(U&& t)
  {
    remaining_space.acquire();
    {
      std::scoped_lock l(items_mtx);
      items.emplace(std::forward<decltype(t)>(t));
    }
    items_produced.release();
  }

  // Enqueue multiple entries.
  template <typename InputIterator>
  void enqueue(InputIterator begin, InputIterator end)
  {
    remaining_space.acquire();
    {
      std::scoped_lock l(items_mtx);
      // TODO: Do I need another overload that moves the items out of the range?
      std::for_each(begin, end, [&] (auto&& item) { items.push(item); });
    }
    items_produced.release(std::distance(begin, end));
  }

  // Attempt to dequeue one entry.
  template <typename Rep, typename Period>
  std::optional<T> try_dequeue_for(
    std::chrono::duration<Rep, Period> const& rel_time
  ) {
    std::optional<T> tmp;
    LOG("entered, about to items_produced.acquire()");
    if (!items_produced.try_acquire_for(rel_time))
      return tmp;
    LOG("items_produced.acquire() succeeded");
    {
      std::scoped_lock l(items_mtx);
      LOG("lock acquired");
      assert(!items.empty());
      tmp = std::move(items.front());
      items.pop();
    }
    LOG("lock released, about to remaining_space.release()");
    remaining_space.release();
    LOG("remaining_space.release() succeeded");
    return tmp; // Do I need to std::move here?
  }

  ~concurrent_bounded_queue() {
    LOG("destroying queue");
  }
};

int main()
{
  std::atomic<int> count(0);

  concurrent_bounded_queue<std::function<void()>, 64> tasks;

  {
    thread_group tg(8,
      [&] (std::stop_token stoken)
      {
        while (!stoken.stop_requested()) {
          auto f = tasks.try_dequeue_for(std::chrono::milliseconds(1));
          if (f) {
            assert(*f);
            (*f)();
          }
        }
        LOG("worker thread exiting");
      }
    );

    for (std::size_t i = 0; i < 256; ++i)
      tasks.enqueue([&] { ++count; });

    std::latch l(9);
    for (std::size_t i = 0; i < 8; ++i)
      tasks.enqueue(
        [&] {
          LOG("arriving at latch");
          l.arrive_and_wait();
          LOG("arrived at latch");
        });
    l.arrive_and_wait();
  }

  // TODO: `format`.
  std::cout << count << std::endl;
}
