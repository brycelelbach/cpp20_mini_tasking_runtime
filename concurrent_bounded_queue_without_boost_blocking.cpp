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

  auto size() { return members.size(); }

  void request_stop() {
    for (auto& t : members) t.request_stop();
  }
};

// TODO: Need a way to shut it down.
// TODO: For forward progress, need to try_acquire and then pop some work from
// the queue when queuing.
// TODO: Should QueueDepth be ptrdiff_t or size_t?

template <typename T, std::size_t QueueDepth>
struct concurrent_bounded_queue {
private:
  std::queue<T> items; // TODO: Would prefer a fixed-sized queue.
  std::mutex items_mtx;
  std::counting_semaphore<QueueDepth> items_produced{0};
  std::counting_semaphore<QueueDepth> remaining_space{QueueDepth};

public:
  constexpr concurrent_bounded_queue() = default;

  ~concurrent_bounded_queue() {
    LOG("destroying queue");
  }

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

  // Dequeue one entry.
  T dequeue() {
    LOG("entered, about to items_produced.acquire()");
    items_produced.acquire();
    std::optional<T> tmp;
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

template <std::size_t QueueDepth>
struct bounded_depth_task_manager
{
private:
  concurrent_bounded_queue<std::function<void()>, QueueDepth> tasks;
  thread_group threads; // This must be the last member initialized in this class;
                        // we start the threads in the class constructor, and the
                        // worker thread function accesses the other members.

  void process_tasks(std::stop_token stoken) {
    while (!stoken.stop_requested())
      tasks.dequeue()();
    LOG("worker thread beginning shutdown");
    // We've gotten a stop request, but there may still be work in the queue,
    // so let's clear it out.
    while (true) {
      if (auto f = tasks.try_dequeue(); f) std::move(*f)();
      else
        break;
    }
    LOG("worker thread has shutdown");
  }

public:
  bounded_depth_task_manager(std::size_t num_threads)
    : threads(num_threads, [&] (std::stop_token stoken) { process_tasks(stoken); })
  {}

  ~bounded_depth_task_manager() {
    std::latch l(threads.size() + 1);
    for (std::size_t i = 0; i < threads.size(); ++i)
      enqueue([&] { l.arrive_and_wait(); });
    threads.request_stop();
    l.arrive_and_wait();
  }

  template <typename Invocable>
  void enqueue(Invocable&& f) {
    tasks.enqueue(std::forward<decltype(f)>(f));
  }
};

void enqueue_tree(auto& tm, std::atomic<std::uint64_t>& count, std::uint64_t level) {
  ++count;
  if (0 != level) {
    tm.enqueue([&tm, &count, level] { enqueue_tree(tm, count, level - 1); });
    tm.enqueue([&tm, &count, level] { enqueue_tree(tm, count, level - 1); });
  }
}

int main()
{
  std::atomic<std::uint64_t> count(0);

  {
    bounded_depth_task_manager<64> tm(8);

    for (std::size_t i = 0; i < 128; ++i)
      tm.enqueue([&] { ++count; });

    enqueue_tree(tm, count, 8);
  }

  // TODO: `format`.
  std::cout << count << "\n";
}
