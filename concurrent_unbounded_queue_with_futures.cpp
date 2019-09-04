#include <queue>
#include <mutex>
#include <variant>
#include <jthread>
#include <stop_token>
#include <atomic_wait>
#include <latch>
#include <iostream>
#include <sstream>
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

// TODO: Coroutines.

template <typename Sig>
struct fire_once;

template <typename T>
struct emplace_as {};

template <typename R, typename... Args>
struct fire_once<R(Args...)>
{
  private:

    std::unique_ptr<void, void(*)(void*)> ptr{nullptr, +[] (void*) {}};
    void(*invoke)(void*, Args...) = nullptr;

  public:

    constexpr fire_once() = default;

    constexpr fire_once(fire_once&&) = default;
    constexpr fire_once& operator=(fire_once&&) = default;

    template <
        typename F
        // {{{ SFINAE
      , std::enable_if_t<!std::is_same_v<std::decay_t<F>, fire_once>, int> = 0
      , std::enable_if_t<
               std::is_convertible_v<
                 std::result_of_t<std::decay_t<F>&(Args...)>, R
               >
            || std::is_same_v<R, void>
          , int
        > = 0
        // }}}
    >
    fire_once(F&& f)
    // {{{
      : fire_once(emplace_as<std::decay_t<F>>{}, std::forward<F>(f))
    {}
    // }}}

    template <typename F, typename...FArgs>
    fire_once(emplace_as<F>, FArgs&&...fargs)
    { // {{{
        auto pf = std::make_unique<F>(std::forward<FArgs>(fargs)...);
        invoke =
            +[](void* pf, Args...args) -> R
            {
                return (*reinterpret_cast<F*>(pf))(std::forward<Args>(args)...);
            };
        ptr = { pf.release(), [] (void* pf) { delete (F*)(pf); } };
    } // }}}

    R operator()(Args... args) &&
    { // {{{
        try {
            if constexpr (std::is_same_v<R, void>)
            {
                invoke(ptr.get(), std::forward<Args>(args)...);
                clear();
            }
            else
            {
                R ret = invoke(ptr.get(), std::forward<Args>(args)...);
                clear();
                return ret;
            }
        } catch (...) {
            clear();
            throw;
        }
    } // }}}

    void clear()
    { // {{{
        invoke = nullptr;
        ptr.reset();
    } // }}}

    explicit operator bool() const
    { // {{{
        return bool(ptr);
    } // }}}
};

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
    return std::move(*tmp);
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
    return std::move(*tmp);
  }
};

struct unbounded_depth_task_manager
{
private:
  concurrent_unbounded_queue<fire_once<void()>> tasks;
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
        std::move(*f)();
        active_task_count.fetch_sub(1, std::memory_order_release);
      }
    }
    LOG("worker thread beginning shutdown");
    // We've gotten a stop request, but there may still be work in the queue,
    // so let's clear it out.
    while (true) {
      auto f = tasks.try_dequeue();
      if (f)
        std::move(*f)();
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

  void boost_block() {
    // Dequeue and execute tasks to make progress.
    auto f = tasks.try_dequeue();
    if (f) {
      active_task_count.fetch_add(1, std::memory_order_release);
      std::move(*f)();
      active_task_count.fetch_sub(1, std::memory_order_release);
    }
  }

  struct executor {
  private:
    unbounded_depth_task_manager* tm;

  public:
    executor(unbounded_depth_task_manager* tm_) : tm(tm_) {
      assert(tm);
    }

    constexpr executor(executor const& other) : tm(other.tm) {
      assert(tm);
    }

    constexpr executor& operator=(executor const& other) {
      tm = other.tm;
      assert(tm);
      return *this; 
    }

    template <typename Invocable>
    void execute(Invocable&& f) {
      assert(tm);
      tm->enqueue(std::forward<decltype(f)>(f));
    }

    void boost_block() {
      assert(tm);
      tm->boost_block();
    }
  };

  auto get_executor() {
    return executor{this};
  }
};

template <typename T>
struct asynchronous_value {
private:
  spin_mutex mtx;
  std::variant<std::monostate, T, fire_once<void(T&&)>> state;

public:
  constexpr asynchronous_value() = default;

  void set_value(T&& t) {
    LOG("setting value");
    fire_once<void(T&&)> tmp;
    {
      std::scoped_lock l(mtx);
      assert(!std::holds_alternative<T>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        LOG("storing value; consumer will execute continuation");
        // We're empty, so store the value.
        state = std::move(t);
      }
      else if (std::holds_alternative<fire_once<void(T&&)>>(state))
        // There's a continuation, so we need to run it.
        tmp = std::move(std::get<fire_once<void(T&&)>>(state));
    }
    if (tmp) {
      LOG("found a continuation; producer will execute it");
      std::move(tmp)(std::move(t));
    }
  }

  template <typename Invocable>
  void set_continuation(Invocable&& f) {
    LOG("setting continuation");
    std::optional<T> tmp;
    {
      std::scoped_lock l(mtx);
      assert(!std::holds_alternative<fire_once<void(T&&)>>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        LOG("storing continuation; producer will execute it");
        // We're empty, so store the continuation.
        state = std::forward<Invocable>(f);
      }
      else if (std::holds_alternative<T>(state))
        // There's a value, so we need to run the continuation.
        tmp = std::move(std::get<T>(state));
    }
    if (tmp) {
      LOG("found a value; consumer will execute continuation");
      std::forward<Invocable>(f)(std::move(*tmp));
    }
  }
};

template <typename T, typename Executor>
struct unique_future;

template <typename T>
struct unique_promise
{
private:
  std::shared_ptr<asynchronous_value<T>> data;
  bool future_retrieved{false}; // If !data, this must be false.

  void deferred_data_allocation() {
    if (!data) {
      assert(!future_retrieved);
      data = std::make_shared<asynchronous_value<T>>();
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
    LOG("retrieving future");
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

  template <typename UExecutor, typename Invocable>
  auto then(UExecutor uexec, Invocable&& f) { 
    assert(data);
    unique_promise<decltype(std::declval<Invocable>()(std::declval<T>()))> p;
    // We have to retrieve the future before we move the promise into the
    // lambda below.
    auto g = p.get_future(uexec);
    data->set_continuation(
      [uexec, f = std::forward<Invocable>(f), p = std::move(p)]
      (T&& v) mutable {
        LOG("enqueuing invocation of continuation");
        uexec.execute(
          [f = std::move(f), p = std::move(p), v = std::move(v)] () mutable  {
            LOG("invoking then continuation");
            p.set(std::invoke(std::move(f), std::move(v)));
          }
        );
      }
    );
    return std::move(g);
  }

  template <typename F>
  auto then(F&& f) {
    return then(exec, std::forward<F>(f));
  }

  auto get() {
    std::counting_semaphore<1> sem;

    std::optional<T> v;

    data->set_continuation(
      [&] (T&& t) {
        LOG("retrieving value");
        v = std::move(t);
        sem.release();
      }
    );

    while (!sem.try_acquire())
      exec.boost_block();

    return std::move(*v);
  }
};

template <typename Executor, typename Invocable, typename... Args>
auto async(Executor exec, Invocable&& f, Args&&... args) {
  unique_promise<decltype(std::declval<Invocable>()(std::declval<Args>()...))> p;
  // We have to retrieve the future before we move the promise into the
  // lambda below.
  auto g = p.get_future(exec);
  exec.execute(
    [f = std::forward<Invocable>(f),
     // We have to either copy or move args into the lambda, as the lambda may
     // outlive the current scope.
     args = std::tuple(std::forward<Args>(args)...),
     p = std::move(p)]
    () mutable {
      p.set(std::apply(std::move(f), std::move(args)));
    }
  );
  return std::move(g);
}

template <typename Executor>
std::uint64_t fibonacci(Executor exec, std::uint64_t n)
{
  std::ostringstream stm;

  stm << "in fibonacci(" << n << ")\n";

  std::cout << stm.str();

  if (n < 2)
    return n;

  auto n1 = async(exec, fibonacci<Executor>, exec, n - 1);
  auto n2 = async(exec, fibonacci<Executor>, exec, n - 2);

  return n1.get() + n2.get();
}

int main()
{
  std::atomic<int> count(0);

  std::uint64_t fib10 = 0;

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

    fib10 = fibonacci(tm.get_executor(), 10);
  }

  std::cout << "fibonacci(10) " << fib10 << "\n";

  // TODO: `format`.
  std::cout << count << std::endl;
}
