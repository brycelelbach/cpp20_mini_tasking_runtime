#include <queue>
#include <vector>
#include <mutex>
#include <variant>
#include <jthread>
#include <stop_token>
#include <atomic_wait>
#include <latch>
#include <barrier>
#include <iostream>
#include <sstream>
#include <semaphore>
#include <functional>
#include <optional>
#include <coroutine>

#if !defined(NDEBUG) && !defined(__NO_LOGGING)
  #define LOG(...)                                                            \
    {                                                                         \
      std::ostringstream stm;                                                 \
      stm << __FUNCTION__ << ": " << __VA_ARGS__ << "\n";                     \
      std::cout << stm.str();                                                 \
    }                                                                         \
    /**/
#else
  #define LOG(...)
#endif

// TODO: Use barrier. Ideas:
//       - Synchronize with outstanding work?
//       - Replace latch in startup code with barrier.

// TODO: Pass executors by value.
// TODO: Remove executors abstraction.

// TODO: Coroutines.

// TODO: Remove unnecessary std::moves.

// TODO: Switch to shared futures for syntax simplicity?

// TODO: const/constexpr/noexcept

// TODO: Reimplement all future stuff in terms of `submit`

// TODO: Make the task manager a sender? That sends itself?

// TODO: Do first chunk in calling thread.

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
    , std::enable_if_t<!std::is_same_v<std::decay_t<F>, fire_once>, int> = 0
    , std::enable_if_t<
             std::is_convertible_v<
               std::result_of_t<std::decay_t<F>&(Args...)>, R
             >
          || std::is_same_v<R, void>
        , int
      > = 0

  >
  fire_once(F&& f)
    : fire_once(emplace_as<std::decay_t<F>>{}, std::forward<F>(f))
  {}


  template <typename F, typename...FArgs>
  fire_once(emplace_as<F>, FArgs&&...fargs)
  {
      auto pf = std::make_unique<F>(std::forward<FArgs>(fargs)...);
      invoke =
          +[](void* pf, Args...args) -> R
          {
              return (*reinterpret_cast<F*>(pf))(std::forward<Args>(args)...);
          };
      ptr = { pf.release(), [] (void* pf) { delete (F*)(pf); } };
  }

  R operator()(Args... args) &&
  {
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
  }

  void clear()
  {
      invoke = nullptr;
      ptr.reset();
  }

  explicit operator bool() const
  {
      return bool(ptr);
  }
};

struct thread_group {
private:
  std::vector<std::jthread> members;

public:
  thread_group(thread_group const&) = delete;
  thread_group& operator=(thread_group const&) = delete;

  template <typename Invocable>
  thread_group(std::size_t count, Invocable&& f) {
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
  fire_once<void()> on_completed;
  bool consumed{false};

public:
  constexpr asynchronous_value() = default;

  bool ready() {
    std::scoped_lock l(mtx);
    return std::holds_alternative<T>(state);
  }

  template <typename U>
  void set_value(U u) {
    LOG(this << ": setting value");
    fire_once<void(T&&)> tmp;
    fire_once<void()> oc;
    {
      std::scoped_lock l(mtx);
      assert(!consumed);
      assert(!std::holds_alternative<T>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        LOG(this << "storing value; consumer will execute continuation");
        // We're empty, so store the value.
        state = std::move(u);
      }
      else if (std::holds_alternative<fire_once<void(T&&)>>(state)) {
        // There's a continuation, so we need to run it.
        tmp = std::move(std::get<fire_once<void(T&&)>>(state));
        consumed = true;
      }
      oc = std::move(on_completed);
    }
    if (tmp) {
      LOG(this << "found a continuation; producer will execute it");
      std::move(tmp)(std::move(u));
    }
    if (oc) {
      LOG(this << "producer running on_completed");
      std::move(oc)();
    }
  }

  template <typename Invocable>
  void set_continuation(Invocable&& f) {
    LOG(this << "setting continuation");
    std::optional<T> tmp;
    fire_once<void()> oc;
    {
      std::scoped_lock l(mtx);
      assert(!consumed);
      assert(!std::holds_alternative<fire_once<void(T&&)>>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        LOG(this << "storing continuation; producer will execute it");
        // We're empty, so store the continuation.
        state = std::forward<Invocable>(f);
      }
      else if (std::holds_alternative<T>(state)) {
        // There's a value, so we need to run the continuation.
        tmp = std::move(std::get<T>(state));
        oc = std::move(on_completed);
        consumed = true;
      }
    }
    if (tmp) {
      LOG(this << "found a value; consumer will execute continuation");
      std::forward<Invocable>(f)(std::move(*tmp));
      if (oc) {
        LOG(this << "consumer running on_completed");
        std::move(oc)();
      }
    }
  }

  template <typename Invocable>
  void set_on_completed(Invocable&& f) {
    LOG(this << "setting on_completed");
    bool run_on_completed = false;
    {
      std::scoped_lock l(mtx);
      assert(!on_completed);
      if (std::holds_alternative<std::monostate>(state)) {
        LOG(this << "storing on_completed; producer will execute it");
        // We're empty, so store the continuation.
        on_completed = std::forward<Invocable>(f);
      }
      else if (std::holds_alternative<T>(state)) {
        // There's a value, so we need to run the continuation.
        run_on_completed = true;
      }
    }
    if (run_on_completed) {
      LOG(this << "found a value; consumer will execute on_completed");
      std::forward<Invocable>(f)();
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
  unique_future<T, std::decay_t<Executor>> get_future(Executor&& exec) {
    LOG("retrieving future");
    deferred_data_allocation();
    future_retrieved = true;
    return unique_future<T, std::decay_t<Executor>>(data, std::forward<Executor>(exec));
  }

  template <typename U>
  void set(U&& u) {
    deferred_data_allocation();
    data->set_value(std::forward<U>(u));
  }
};

template <typename T, typename Executor>
T unique_future_await_resume(unique_future<T, Executor>& f) {
  assert(f.ready());
  return *f.try_get();
}

template <typename T, typename Executor0, typename Executor1>
T unique_future_await_resume(unique_future<unique_future<T, Executor0>, Executor1>& f) {
  assert(f.ready());
  auto g = *f.try_get();
  assert(g.ready());
  return *std::move(g).try_get();
}

template <typename T>
struct is_unique_future : std::false_type {};

template <typename T, typename Executor>
struct is_unique_future<unique_future<T, Executor>> : std::true_type {};

template <typename T>
inline constexpr bool is_unique_future_v = is_unique_future<T>::value;

template <typename T, typename Executor>
struct unique_future
{
  using value_type = T;
  using executor_type = Executor;

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

  template <typename U, typename UExecutor>
  friend struct unique_future;

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
      return data->ready();
    return false;
  }

  template <typename UExecutor, typename Invocable>
  auto submit(UExecutor uexec, Invocable&& f) {
    data->set_continuation(
      [uexec, f = std::forward<Invocable>(f)]
      (T&& t) mutable {
        LOG("enqueuing invocation of continuation");
        uexec.execute(
          [f = std::move(f), t = std::move(t)] () mutable  {
            LOG("invoking then continuation");
            std::invoke(std::move(f), std::move(t));
          }
        );
      }
    );
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
      (T&& t) mutable {
        LOG("enqueuing invocation of continuation");
        uexec.execute(
          [f = std::move(f), p = std::move(p), t = std::move(t)] () mutable  {
            LOG("invoking then continuation");
            p.set(std::invoke(std::move(f), std::move(t)));
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

  // TODO: Should this be rvalue-ref qualified?
  template <typename UExecutor>
  auto unwrap(UExecutor uexec) {
    if constexpr (is_unique_future_v<T>) {
      // TODO: Rename V to U if we take out executors.
      using V = typename T::value_type;
      using VExecutor = typename T::executor_type;

      unique_promise<V> p;
      // We have to retrieve the future before we move the promise into the
      // lambda below.
      auto g = p.get_future(uexec);
      data->set_continuation(
        [uexec, p = std::move(p)] (unique_future<V, VExecutor>&& in) mutable {
          uexec.execute(
            [in = std::move(in), p = std::move(p)] () mutable {
              in.data->set_continuation(
                [in = std::move(in), p = std::move(p)] (V&& v) mutable {
                  in.executor().execute(
                    [p = std::move(p), v = std::move(v)] () mutable {
                      p.set(std::move(v));
                    }
                  );
                }
              );
            }
          );
        }
      );
      return std::move(g);
    } else
      return std::move(*this);
  }

  auto unwrap() {
    return unwrap(exec);
  }

  std::optional<T> try_get() {
    if (ready()) {
      std::optional<T> v;

      // Since the data is ready, this will run immediately in this thread.
      data->set_continuation([&] (T&& t) { v = std::move(t); });

      assert(v);
      return std::move(*v);
    }

    return {};
  }

  T get() {
    std::counting_semaphore<1> sem;

    std::optional<T> v;

    // TODO: Should this be using `then`?
    data->set_continuation(
      [&] (T&& t) {
        LOG("retrieving value");
        v = std::move(t);
        sem.release();
      }
    );

    while (!sem.try_acquire())
      exec.boost_block();

    assert(v);
    return std::move(*v);
  }

  bool await_ready() {
    return ready();
  }

  template <typename Promise>
  void await_suspend(std::experimental::coroutine_handle<Promise> c) {
    data->set_on_completed(
      [exec = exec, c = std::move(c)] () mutable {
        exec.execute(
          [c = std::move(c)] () mutable {
            c();
          }
        );
      }
    );
  }

  auto await_resume() {
    return unique_future_await_resume(*this);
  }

  Executor executor() {
    return exec;
  }
};

namespace std { namespace experimental {

// TODO: Merge with promise when we nuke executors in this example.
// TODO: Combine set and return_value.

template <typename T, typename Executor, typename... Args>
struct coroutine_traits<unique_future<T, Executor>, Executor, Args...>
{
  struct promise_type : unique_promise<T> {
    using executor_type = std::decay_t<Executor>;

  private:
    executor_type exec;

  public:
    template <typename UExecutor, typename... UArgs>
    promise_type(UExecutor uexec, UArgs&&...) : exec(uexec) {}

    unique_future<T, executor_type> get_return_object() {
      return this->get_future(exec);
    }

    std::experimental::suspend_never initial_suspend() { return {}; }

    std::experimental::suspend_never final_suspend() { return {}; }

    template <typename U>
    void return_value(U&& u) {
      LOG("setting return_value");
      this->set(std::forward<U>(u));
    }

    void unhandled_exception() { throw; }
  };
};

}}

template <typename Executor, typename T>
auto ready_future(Executor exec, T&& t) {
  unique_promise<std::decay_t<T>> p;
  p.set(std::forward<T>(t));
  return p.get_future(exec);
}

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
  return g.unwrap();
}

template <typename T, typename Executor0, typename Executor1>
auto when_all(Executor0 exec, std::vector<unique_future<T, Executor1>>& fs) {
  // FIXME: Only works for default constructible types.
  auto results = std::make_shared<std::vector<T>>(fs.size());

  unique_promise<std::vector<T>> p;
  auto g = p.get_future(exec);

  auto finalize =
    [exec, results, p = std::move(p)] () mutable {
      exec.execute(
        [results, p = std::move(p)] () mutable {
          LOG("finalizing barrier");
          p.set(std::move(*results));
        }
      );
    };

  auto barrier = std::make_shared<std::barrier<decltype(finalize)>>(
    fs.size(), std::move(finalize)
  );

  for (std::size_t i = 0; i < fs.size(); ++i) {
    fs[i].submit(
      exec,
      [i, results, barrier] (T&& t) mutable {
        LOG("assigning when_all result[" << i << "]");
        (*results)[i] = std::move(t);
        barrier->arrive_and_drop();
      }
    );
  }

  return g;
}

template <typename Executor>
unique_future<std::uint64_t, Executor> afibonacci(Executor exec, std::uint64_t n)
{
  LOG("in afibonacci(" << n << ")");

  if (n >= 2)
    n = co_await afibonacci(exec, n - 1)
      + co_await async(exec, afibonacci<Executor>, exec, n - 2);
  co_return n;
}

namespace std {

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOp>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init, BinaryOp op)
{
  if (first != last) {
    T saved = init;
    do {
      init = op(init, *first);
      *result = saved;
      saved = init;
      ++result;
    } while (++first != last);
  }
  return result;
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init)
{
  return exclusive_scan(first, last, result, init, std::plus{});
}

template <typename InputIterator, typename OutputIterator, typename BinaryOp, typename T>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOp op, T init)
{
  for (; first != last; ++first, (void) ++result) {
    init = op(init, *first);
    *result = init;
  }
  return result;
}

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOp op)
{
  if (first != last) {
    typename std::iterator_traits<InputIterator>::value_type init = *first;
    *result++ = init;
    if (++first != last)
      return inclusive_scan(first, last, result, op, init);
  }

  return result;
}

} // namespace std

// TODO: Test this for size = 0, size = 1, size = 2, etc.
// TODO: Test this for small chunk sizes.
template <typename Executor, typename InputIt, typename OutputIt, typename BinaryOp, typename T>
unique_future<OutputIt, Executor>
async_exclusive_scan(Executor exec, InputIt first, InputIt last, OutputIt output,
                     T init, BinaryOp op, std::size_t chunk_size)
{
  std::size_t const elements = std::distance(first, last);
  std::size_t const chunks   = (1 + ((elements - 1) / chunk_size)); // Round up.

  std::vector<unique_future<T, Executor>> sweep;
  sweep.reserve(chunks);

  // Upsweep.
  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec,
      [=] {
        auto const this_begin = chunk * chunk_size;
        auto const this_end   = std::min(elements, (chunk + 1) * chunk_size);
        LOG("upsweep (" << this_begin << ", " << this_end << ")");
        // Save the value of the final element which we need to add into the
        // sum we return, just in case this is an in-place scan.
        auto const last_element = first[this_end - 1];
        // Add the value of the final element into the sum we return.
        // FIXME: Probably wrong for small chunk sizes.
        return op(*--std::exclusive_scan(first + this_begin, first + this_end,
                                         output + this_begin, T{}, op),
                  last_element);
      }
    ));

  auto sums = co_await when_all(exec, sweep);

  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    LOG("sums[" << chunk << "] == " << sums[chunk]);

  // We add in init here.
  std::inclusive_scan(sums.begin(), sums.end(), sums.begin(), op, init);

  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    LOG("scanned sums[" << chunk << "] == " << sums[chunk]);

  sweep.clear();

  // Downsweep.
  for (std::size_t chunk = 1; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec,
      // Taking sums by reference is fine here; we're going to co_await on the
      // futures, which will keep the current scope alive for them.
      [=, &sums] {
        auto const this_begin = chunk * chunk_size;
        auto const this_end   = std::min(elements, (chunk + 1) * chunk_size);
        LOG("downsweep (" << this_begin << ", " << this_end << ")");
        std::for_each(output + this_begin, output + this_end,
                      [=, &sums] (auto& t) {
                        LOG("downsweep for_each t(" << t << ") sums(" << sums[chunk - 1] << ")");
                        t = op(std::move(t), sums[chunk - 1]);
                      });
        return T(*(output + this_end - 1));
      }
    ));

  co_await when_all(exec, sweep);

  co_return output + elements;
}

// How is this not a thing?
namespace std {

template <typename ForwardIt, typename Size, typename T>
ForwardIt iota_n(ForwardIt first, Size count, T value) {
  for (Size i = 0; i < count; ++i)
    *first++ = value++;
  return first;
}

} // namespace std

int main()
{
  std::vector<std::uint64_t> u;

  std::fill_n(std::back_inserter(u), 128, 1);

  for (std::size_t i = 0; i < u.size(); ++i)
    std::cout << "initial u[" << i << "]: " << u[i] << "\n";

  std::exclusive_scan(u.begin(), u.end(), u.begin(), 0, std::plus{});

  for (std::size_t i = 0; i < u.size(); ++i)
    std::cout << "gold u[" << i << "]: " << u[i] << "\n";

  u.clear();
  std::fill_n(std::back_inserter(u), 128, 1);

  std::vector<std::uint64_t> v = {
    1, 0, 1, 0, 1, 1, 1, 1,
    1, 0, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 1, 0, 1
  };

  for (std::size_t i = 0; i < u.size(); ++i)
    std::cout << "initial v[" << i << "]: " << v[i] << "\n";

  std::exclusive_scan(v.begin(), v.end(), v.begin(), 0, std::plus{});

  for (std::size_t i = 0; i < u.size(); ++i)
    std::cout << "gold v[" << i << "]: " << v[i] << "\n";

  v.clear();
  v = std::vector<std::uint64_t>{
    1, 0, 1, 0, 1, 1, 1, 1,
    1, 0, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 1, 0, 1
  };

  {
    unbounded_depth_task_manager tm(8);

    async_exclusive_scan(tm.get_executor(),
                         u.begin(), u.end(), u.begin(), std::uint64_t{0}, std::plus{},
                         4).get();

    async_exclusive_scan(tm.get_executor(),
                         v.begin(), v.end(), v.begin(), std::uint64_t{0}, std::plus{},
                         4).get();
  }

  for (std::size_t i = 0; i < u.size(); ++i)
    std::cout << "observed u[" << i << "]: " << u[i] << "\n";

  for (std::size_t i = 0; i < v.size(); ++i)
    std::cout << "observed v[" << i << "]: " << v[i] << "\n";
}

