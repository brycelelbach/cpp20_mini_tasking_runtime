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

  void request_stop() {
    for (auto& t : members) t.request_stop();
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

  // TODO: Lift common queue mutual exclusion code from enqueue variants.

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

  // Attempt to enqueue one entry.
  template <typename U>
  bool try_enqueue(U&& t)
  {
    if (!remaining_space.try_acquire())
      return false;
    {
      std::scoped_lock l(items_mtx);
      items.emplace(std::forward<decltype(t)>(t));            
    }
    items_produced.release();
    return true;
  }

  // TODO: Lift common queue mutual exclusion code from dequeue variants.
  
  // Attempt to dequeue one entry.
  std::optional<T> try_dequeue() {
    std::optional<T> tmp;
    LOG("entered, about to items_produced.acquire()");
    if (!items_produced.try_acquire())
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
    return *tmp; // Do I need to std::move here?
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
    return *tmp; // Do I need to std::move here?
  }
};

int main() 
{
  std::atomic<int> count(0);

  concurrent_bounded_queue<std::function<void()>, 64> tasks;

  {
    std::atomic<int> active(0);

    std::latch exit(5);
    thread_group tg(4,
      [&] (std::stop_token stoken)
      {
        while (!stoken.stop_requested()) {
          auto f = tasks.try_dequeue_for(std::chrono::milliseconds(1));
          if (f) {
            active.fetch_add(1, std::memory_order_release);
            (*f)();
            active.fetch_sub(1, std::memory_order_release);
          }
        }
        LOG("worker thread beginning shutdown");
        while (true) {
          auto f = tasks.try_dequeue();
          if (f)
            (*f)();
          else if (0 == active.load(std::memory_order_acquire))
            break;
        }
        LOG("worker thread has shutdown; arriving at latch");
        exit.arrive_and_wait();
        LOG("worker thread has shutdown; arrived at latch");
      }
    );

    auto enqueue_boost_block =
      [&] (auto&& f) {
        while (!tasks.try_enqueue(std::forward<decltype(f)>(f))) { 
          auto f = tasks.try_dequeue();
          if (f) {
            active.fetch_add(1, std::memory_order_release);
            (*f)();
            active.fetch_sub(1, std::memory_order_release);
          }
        }
      };

    for (std::size_t i = 0; i < 128; ++i)
      tasks.enqueue([&] { ++count; });

    struct enqueue_recursively {
      decltype(enqueue_boost_block)& enqueue_boost_block;
      decltype(tasks)& tasks;
      decltype(count)& count;
      std::size_t levels_remaining;

      void operator()() {
        ++count;
        if (0 == levels_remaining) return;
        enqueue_boost_block(enqueue_recursively{enqueue_boost_block, tasks, count, levels_remaining - 1});
        enqueue_boost_block(enqueue_recursively{enqueue_boost_block, tasks, count, levels_remaining - 1});
      }
    };

    tasks.enqueue(enqueue_recursively{enqueue_boost_block, tasks, count, 8});

    tg.request_stop();
    exit.arrive_and_wait();
  }

  // TODO: `format`.
  std::cout << count << std::endl;
}
