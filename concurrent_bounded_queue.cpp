#include <queue>
#include <mutex>
#include <jthread>
#include <stop_token>
#include <latch>
#include <iostream>
#include <semaphore>
#include <optional>

#include <cjdb/concepts.hpp>

namespace std { using namespace cjdb; }

struct thread_group {
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
    for (auto& thread : members) thread.request_stop();
  }

  std::vector<std::jthread> members;
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

  // Dequeue one entry.
  T dequeue()
  {
    std::cout << "entered dequeue\n";
    items_produced.acquire();
    std::cout << "items_produced.acquire() succeeded\n";
    std::optional<T> tmp;
    {
      std::scoped_lock l(items_mtx);
      std::cout << "lock acquired\n";
      tmp = std::move(items.front());
      items.pop();
    }
    std::cout << "releasing space\n";
    remaining_space.release();
    std::cout << "exiting dequeue\n";
    return *tmp; // Do I need to std::move here?
  }
};

int main() 
{
  std::atomic<int> count(0);

  concurrent_bounded_queue<std::function<void()>, 64> tasks;

  {
    thread_group tg(4,
      [&] (std::stop_token stop)
      {
        std::function<void()> f;
        do {
          tasks.dequeue()();
        } while (!stop.stop_requested());
        std::cout << "exiting\n";
      }
    );

    tg.request_stop(); 

    std::latch l(5);
    for (std::size_t i = 0; i < 4; ++i)
      tasks.enqueue([&] { 
        std::cout << "arriving at latch\n";
        l.arrive_and_wait();
      });
    l.arrive_and_wait();

  }

  // TODO: `format`.
  std::cout << count << std::endl;
}
