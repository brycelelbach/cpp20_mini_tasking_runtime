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

// TODO: Pass executors by value.

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

struct spin_mutex {
private:
	std::atomic<bool> flag = ATOMIC_VAR_INIT(0);

public:
	void lock() noexcept {
		while (1 == flag.exchange(1, std::memory_order_acquire))
			atomic_wait_explicit(&flag, 1, std::memory_order_relaxed);
	}

	void unlock() noexcept {
		flag.store(0, std::memory_order_release);
		atomic_notify_one(&flag);
	}
};

template <typename T>
struct concurrent_unbounded_queue {
private:
  std::queue<T> items; 
  spin_mutex items_mtx;
  std::counting_semaphore<> items_produced{0};

public:
  constexpr concurrent_unbounded_queue() = default;

  ~concurrent_unbounded_queue() {
    LOG("destroying queue");
  }

  // TODO: Lift common queue mutual exclusion code from enqueue variants.

  // Enqueue one entry.
  template <typename U>
  void enqueue(U&& t)
  {
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
    {
      std::scoped_lock l(items_mtx);
      // TODO: Do I need another overload that moves the items out of the range?
      std::for_each(begin, end, [&] (auto&& item) { items.push(item); });
    }
    items_produced.release(std::distance(begin, end));
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
    return *tmp; // Do I need to std::move here?
  }
};

struct unbounded_depth_task_manager
{
private:
  concurrent_unbounded_queue<std::function<void()>> tasks;
  std::atomic<std::size_t> active_task_count{0};
  std::latch exit_latch;
  thread_group threads; // This must be the last member initialized in this class;
                        // we start the threads in the class constructor, and the
                        // worker thread function accesses the other members.

  void process_tasks(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
      auto f = tasks.try_dequeue_for(std::chrono::milliseconds(1));
      if (f) {
        active_task_count.fetch_add(1, std::memory_order_release);
        (*f)();
        active_task_count.fetch_sub(1, std::memory_order_release);
      }
    }
    LOG("worker thread beginning shutdown");
    // We've gotten a stop request, but there may still be work in the queue,
    // so let's clear it out.
    while (true) {
      auto f = tasks.try_dequeue();
      if (f)
        (*f)();
      else if (0 == active_task_count.load(std::memory_order_acquire))
        break;
    }
    LOG("worker thread has shutdown; arriving at latch");
    exit_latch.arrive_and_wait();
    LOG("worker thread has shutdown; arrived at latch");
  }

public:
  unbounded_depth_task_manager(std::size_t num_threads)
    : exit_latch(num_threads + 1)
    , threads(num_threads, [&] (std::stop_token stoken) { process_tasks(stoken); })
  {}

  ~unbounded_depth_task_manager() {
    threads.request_stop();
    exit_latch.arrive_and_wait();
  }

  template <typename Invocable>
  void enqueue(Invocable&& f) {
    tasks.enqueue(std::forward<decltype(f)>(f));
  }

  auto get_executor() {
    struct executor {
      unbounded_depth_task_manager& tm;

      template <typename Invocable>
      void execute(Invocable&& f) {
        tm.enqueue(std::forward<decltype(f)>(f));
      }
    };
    return executor{tm};
  }
};

template <typename T>
struct asynchronous_value {
private:
  spin_mutex mtx;
  std::variant<std::monostate, T, std::function<void(T)>> state;

public:
  constexpr asynchronous_value() = default;

  template <typename U>
  void set_value(U&& u) {
    std::function<void(T)> tmp;
    {
      std::scoped_lock l(mtx);
      assert(!std::holds_alternative<T>(state));
      if (std::holds_alternative<std::monostate>(state))
        // We're empty, so store the value.
        data = std::forward<U>(u);
      else if (std::holds_alternative<std::function<void(T)>>(state))
        // There's a continuation, so we need to run it.
        tmp = std::move(std::get<std::function<void(T)>>(tmp));
    }
    if (tmp)
      tmp(std::forward<U>(u));
  }

  template <typename Invocable>
  void set_continuation(Invocable&& f) {
    std::optional<T> tmp;
    {
      std::scoped_lock l(mtx);
      assert(!std::holds_alternative<std::function<void(T)>>(state));
      if (std::holds_alternative<std::monostate>(data))
        // We're empty, so store the continuation.
        data = std::forward<Invocable>(f);
      else if (std::holds_alternative<T>(state))
        // There's a value, so we need to run the continuation.
        tmp = std::move(std::get<T>(tmp));
    }
    if (tmp)
      std::forward<Invocable>(f)(*tmp);
  }
};

template <typename T, typename Executor>
struct unique_future;

template <typename T>
struct unique_promise
{
private:
  std::shared_ptr<asynchronous_value<T>> data;
  bool future_retrieved; // If !data, this must be false.

  void deferred_data_allocation() {
    if (!data) {
      assert(!future_retrieved);
      data = std::make_shared<data_type>();
    }
  }

public:
  // DefaultConstructible
  constexpr unique_promise() noexcept = default;

  // MoveAssignable 
  constexpr unique_promise(unique_promise&&) noexcept = default;
  constexpr unique_promise& operator=(unique_promise&&) noexcept = default;

  // Not CopyAssignable
  unique_promise(unique_promise const&) = delete;
  unique_promise& operator=(unique_promise const&) = delete;

  bool ready() const {
    if (data)
      return data->value_ready();
    return false;
  }

  // Precondition: Future has not been retrieved yet.
  template <typename Executor>
  unique_future<T, Executor> get_future(Executor&& exec) { 
    deferred_data_allocation();
    future_retrieved = true; 
    return unique_future<T, Executor>(data, std::forward<Executor>(exec));
  }

  template <typename U>
  void set(U&& u) {
    deferred_data_allocation();
    data->set_value(std::forward<U>(u));
  }
};

template <typename T, typename Executor>
struct unique_future
{
private:
  std::shared_ptr<asynchronous_value<T>> data;
  Executor exec;

  template <typename UExecutor>
  unique_future(
    std::shared_ptr<asynchronous_value<T>> ptr, UExecutor&& uexec
  )
    : data(ptr)
    , exec(std::forward<UExecutor>(uexec))
  {}

  friend struct unique_promise<T>;

public:
  // DefaultConstructible
  constexpr unique_future() noexcept = default;

  // MoveAssignable 
  constexpr unique_future(unique_future&&) noexcept = default;
  constexpr unique_future& operator=(unique_future&&) noexcept = default;

  // Not CopyAssignable
  unique_future(unique_future const&) = delete;
  unique_future& operator=(unique_future const&) = delete;

  bool ready() const { 
    if (data)
      return data->value_ready();
    return false;
  } 

  template <typename UExecutor, typename F>
  auto then(UExecutor&& uexec, F&& f) { 
    assert(data);
    unique_promise<decltype(std::declval<F>()(std::declval<T>()))> p;
    data->set_continuation(
      [uexec = std::forward<UExecutor>(uexec),
       f = std::forward<F>(f), 
       p = std::move(p)]
      (T v) mutable {
        std::move(uexec).execute(
          [f = std::move(f), p = std::move(p)] mutable {
            p.set(std::apply(std::move(f), std::move(v)));
          }
        );
      }
    );
    return p.get_future();
  }

  template <typename F>
  auto then(F&& f) {
    return then(exec, std::forward<F>(f));
  }
};

template <typename Executor, typename Invocable, typename... Args>
auto async(Executor&& exec, Invocable&& f, Args&&... args) {
  unique_promise<decltype(std::declval<F>()(std::declval<Args>()...))> p;
  std::forward<Executor>(exec).execute(
    [f = std::forward<Invocable>(f),
     args = std::forward_as_tuple(std::forward<Args>(args)...)]
    mutable {
      std::apply(std::move(f), std::move(args));
    }
  );
  return p.get_future();
}

int main()
{
  std::atomic<int> count(0);

  {
    unbounded_depth_task_manager tm(8);

    for (std::size_t i = 0; i < 128; ++i)
      tm.enqueue([&] { ++count; });

    struct enqueue_recursively {
      decltype(tm)& tm;
      decltype(count)& count;
      std::size_t levels_remaining;

      void operator()() {
        ++count;
        if (0 != levels_remaining) {
          tm.enqueue(enqueue_recursively{tm, count, levels_remaining - 1});
          tm.enqueue(enqueue_recursively{tm, count, levels_remaining - 1});
        }
      }
    };

    tm.enqueue(enqueue_recursively{tm, count, 8});
  }

  // TODO: `format`.
  std::cout << count << std::endl;
}
