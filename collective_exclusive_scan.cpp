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
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdlib>

#if !defined(NDEBUG) && !defined(__NO_TASKING_LOGGING)
  #define TASKLOG(...)                                                        \
    {                                                                         \
      std::ostringstream stm;                                                 \
      stm << "tasking: "                                                      \
          << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": "       \
          << __VA_ARGS__ << "\n";                                             \
      std::cout << stm.str();                                                 \
    }                                                                         \
    /**/
#else
  #define TASKLOG(...)
#endif

#if !defined(NDEBUG) && !defined(__NO_SYNCHRONIZATION_LOGGING)
  #define SYNCLOG(...)                                                        \
    {                                                                         \
      std::ostringstream stm;                                                 \
      stm << "synchronization: "                                              \
          << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": "       \
          << __VA_ARGS__ << "\n";                                             \
      std::cout << stm.str();                                                 \
    }                                                                         \
    /**/
#else
  #define SYNCLOG(...)
#endif

#if !defined(NDEBUG) && !defined(__NO_ALGORITHM_LOGGING)
  #define ALGOLOG(...)                                                        \
    {                                                                         \
      std::ostringstream stm;                                                 \
      stm << "algorithm: "                                                    \
          << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": "       \
          << __VA_ARGS__ << "\n";                                             \
      std::cout << stm.str();                                                 \
    }                                                                         \
    /**/
#else
  #define ALGOLOG(...)
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
  thread_group(std::uint64_t count, Invocable&& f) {
    // TODO: Something something ranges, something something no raw loops.
    members.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
      members.emplace_back(std::jthread(
        [=] (std::stop_token stoken) { f(i, stoken); }
      ));
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
    TASKLOG("destroying queue");
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
    TASKLOG("entered, about to items_produced.acquire()");
    if (!items_produced.try_acquire())
      return tmp;
    TASKLOG("items_produced.acquire() succeeded");
    {
      std::scoped_lock l(items_mtx);
      TASKLOG("lock acquired");
      assert(!items.empty());
      tmp = std::move(items.front());
      items.pop();
    }
    TASKLOG("lock released, about to remaining_space.release()");
    return std::move(*tmp);
  }

  // Attempt to dequeue one entry.
  template <typename Rep, typename Period>
  std::optional<T> try_dequeue_for(
    std::chrono::duration<Rep, Period> const& rel_time
  ) {
    std::optional<T> tmp;
    TASKLOG("entered, about to items_produced.acquire()");
    if (!items_produced.try_acquire_for(rel_time))
      return tmp;
    TASKLOG("items_produced.acquire() succeeded");
    {
      std::scoped_lock l(items_mtx);
      TASKLOG("lock acquired");
      assert(!items.empty());
      tmp = std::move(items.front());
      items.pop();
    }
    TASKLOG("lock released, about to remaining_space.release()");
    return std::move(*tmp);
  }
};

struct work_stealing_task_manager
{
private:
  static thread_local std::uint32_t this_thread_index;

  std::vector<concurrent_unbounded_queue<fire_once<void()>>> tasks;
  std::vector<std::uint32_t> enqueue_indices;
  std::vector<std::uint32_t> dequeue_indices;

  std::atomic<std::uint64_t> active_task_count{0};
  std::latch exit_latch;

  thread_group threads; // This must be the last member initialized in this class;
                        // we start the threads in the class constructor, and the
                        // worker thread function accesses the other members.

  std::uint32_t advance_index(std::uint32_t& idx) {
    std::uint32_t tmp = idx;
    idx = (idx + 1) % tasks.size();
    return tmp;
  }

  std::uint32_t next_enqueue_index() {
    return advance_index(enqueue_indices[this_thread_index]);
  }

  std::uint32_t next_dequeue_index() {
    return advance_index(dequeue_indices[this_thread_index]);
  }

  void reset_dequeue_index() {
    dequeue_indices[this_thread_index] = this_thread_index;
  }

  void process_tasks(std::stop_token stoken) {
    TASKLOG(this_thread_index
            << ": worker thread started; entering primary work loop");
    while (!stoken.stop_requested()) {
      auto f = tasks[next_dequeue_index()].try_dequeue();
      if (f) {
        reset_dequeue_index();
        active_task_count.fetch_add(1, std::memory_order_release);
        std::move(*f)();
        active_task_count.fetch_sub(1, std::memory_order_release);
      }
    }
    TASKLOG(this_thread_index
            << ": worker thread beginning shutdown");
    // We've gotten a stop request, but there may still be work in the queue,
    // so let's clear it out.
    while (true) {
      auto f = tasks[this_thread_index].try_dequeue();
      if (f) {
        active_task_count.fetch_add(1, std::memory_order_release);
        std::move(*f)();
        active_task_count.fetch_sub(1, std::memory_order_release);
      }
      else if (0 == active_task_count.load(std::memory_order_acquire))
        break;
    }
    TASKLOG(this_thread_index
            << ": worker thread has shutdown; arriving at latch");
    exit_latch.arrive_and_wait();
    TASKLOG(this_thread_index
            << ": worker thread has shutdown; arrived at latch");
  }

public:
  work_stealing_task_manager(std::uint64_t num_threads)
    : tasks(num_threads + 1)
    , enqueue_indices(num_threads + 1, 0)
    , dequeue_indices(num_threads + 1, 0)
    , exit_latch(num_threads + 1)
    , threads(num_threads,
        [&] (std::uint64_t thread_index, std::stop_token stoken)
        {
          this_thread_index = thread_index;
          process_tasks(stoken);
        })
  {}

  ~work_stealing_task_manager() {
    // We better be destroying the task manager from an external thread, not
    // from inside the system.
    assert(0 == this_thread_index);
    TASKLOG("sending stop request to all threads");
    threads.request_stop();
    TASKLOG("clearing out queue 0");
    // Clear the "external" queue.
    while (true) {
      auto f = tasks[this_thread_index].try_dequeue();
      if (f)
        std::move(*f)();
      else break;
    }
    TASKLOG("task manager has shutdown; arriving at latch");
    exit_latch.count_down();//arrive_and_wait();
    TASKLOG("task manager has shutdown; arrived at latch");
  }

  template <typename Invocable>
  void submit(Invocable&& f) {
    tasks[next_enqueue_index()].enqueue(std::forward<decltype(f)>(f));
  }

  void boost_block() {
    // Dequeue and execute tasks to make progress.
    auto f = tasks[next_dequeue_index()].try_dequeue();
    if (f) {
      reset_dequeue_index();
      std::move(*f)();
    }
  }
};

thread_local std::uint32_t work_stealing_task_manager::this_thread_index{0};

template <typename Submitter>
struct execution_policy_with_chunk_size {
private:
  Submitter& s;
  std::uint64_t cs;

public:
  execution_policy_with_chunk_size(Submitter& s_, std::uint64_t cs_)
    : s(s_), cs(cs_)
  {}

  Submitter& submitter() { return s; }

  std::uint64_t chunk_size() { return cs; }
};

template <typename Submitter>
auto execution_policy(Submitter& s, std::uint64_t cs) {
  return execution_policy_with_chunk_size(s, cs);
}

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
    SYNCLOG(this << ": setting value");
    fire_once<void(T&&)> tmp;
    fire_once<void()> oc;
    {
      std::scoped_lock l(mtx);
      assert(!consumed);
      assert(!std::holds_alternative<T>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        SYNCLOG(this << ": storing value; consumer will execute continuation");
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
      SYNCLOG(this << ": found a continuation; producer will execute it");
      std::move(tmp)(std::move(u));
    }
    if (oc) {
      SYNCLOG(this << ": producer running on_completed");
      std::move(oc)();
    }
  }

  template <typename Invocable>
  void set_continuation(Invocable&& f) {
    SYNCLOG(this << ": setting continuation");
    std::optional<T> tmp;
    fire_once<void()> oc;
    {
      std::scoped_lock l(mtx);
      assert(!consumed);
      assert(!std::holds_alternative<fire_once<void(T&&)>>(state));
      if (std::holds_alternative<std::monostate>(state)) {
        SYNCLOG(this << ": storing continuation; producer will execute it");
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
      SYNCLOG(this << ": found a value; consumer will execute continuation");
      std::forward<Invocable>(f)(std::move(*tmp));
      if (oc) {
        SYNCLOG(this << ": consumer running on_completed");
        std::move(oc)();
      }
    }
  }

  template <typename Invocable>
  void set_on_completed(Invocable&& f) {
    SYNCLOG(this << ": setting on_completed");
    bool run_on_completed = false;
    {
      std::scoped_lock l(mtx);
      assert(!on_completed);
      if (std::holds_alternative<std::monostate>(state)) {
        SYNCLOG(this << ": storing on_completed; producer will execute it");
        // We're empty, so store the continuation.
        on_completed = std::forward<Invocable>(f);
      }
      else if (std::holds_alternative<T>(state)) {
        // There's a value, so we need to run the continuation.
        run_on_completed = true;
      }
    }
    if (run_on_completed) {
      SYNCLOG(this << ": found a value; consumer will execute on_completed");
      std::forward<Invocable>(f)();
    }
  }
};

template <typename T>
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
  constexpr unique_promise() = default;

  // MoveAssignable
  constexpr unique_promise(unique_promise&&) = default;
  constexpr unique_promise& operator=(unique_promise&&) = default;

  // Not CopyAssignable
  unique_promise(unique_promise const&) = delete;
  unique_promise& operator=(unique_promise const&) = delete;

  bool ready() const {
    if (data) return data->ready();
    return false;
  }

  // Precondition: Future has not been retrieved yet.
  unique_future<T> get_return_object() {
    SYNCLOG("retrieving future");
    deferred_data_allocation();
    future_retrieved = true;
    return unique_future<T>(data);
  }

  template <typename U>
  void return_value(U&& u) {
    deferred_data_allocation();
    data->set_value(std::forward<U>(u));
  }

  std::experimental::suspend_never initial_suspend() { return {}; }

  std::experimental::suspend_never final_suspend() { return {}; }

  void unhandled_exception() { throw; }
};

template <typename T>
struct is_unique_future : std::false_type {};

template <typename T>
struct is_unique_future<unique_future<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_unique_future_v = is_unique_future<T>::value;

template <typename T>
struct unique_future
{
  using value_type = T;
  using promise_type = unique_promise<T>;

private:
  std::shared_ptr<asynchronous_value<T>> data;

  unique_future(std::shared_ptr<asynchronous_value<T>> ptr)
    : data(ptr)
  {}

  template <typename U>
  friend unique_future<std::vector<U>> when_all(std::vector<unique_future<U>>& fs);

  friend struct unique_promise<T>;

  template <typename U>
  friend struct unique_future;

public:
  // DefaultConstructible
  constexpr unique_future() = default;

  // MoveAssignable
  constexpr unique_future(unique_future&&) = default;
  constexpr unique_future& operator=(unique_future&&) = default;

  // Not CopyAssignable
  unique_future(unique_future const&) = delete;
  unique_future& operator=(unique_future const&) = delete;

  template <typename U>
  unique_future(U&& u)
    : data(std::make_shared<asynchronous_value<T>>()) {
    data->set_value(std::forward<U>(u));
  }

  template <typename Invocable>
  void submit(Invocable&& f) {
    data->set_continuation(
      [f = std::forward<Invocable>(f)]
      (T&& t) mutable {
        SYNCLOG("invoking then continuation");
        std::invoke(std::move(f), std::move(t));
      }
    );
  }

  template <typename Invocable>
  auto then(Invocable&& f) {
    assert(data);
    unique_promise<decltype(std::declval<Invocable>()(std::declval<T>()))> p;
    // We have to retrieve the future before we move the promise into the
    // lambda below.
    auto g = p.get_return_object();
    data->set_continuation(
      [f = std::forward<Invocable>(f), p = std::move(p)]
      (T&& t) mutable {
        SYNCLOG("enqueuing invocation of continuation");
        p.return_value(std::invoke(std::move(f), std::move(t)));
      }
    );
    return std::move(g);
  }

  auto unwrap() {
    if constexpr (is_unique_future_v<T>) {
      using U = typename T::value_type;
      unique_promise<U> p;
      // We have to retrieve the future before we move the promise into the
      // lambda below.
      auto g = p.get_return_object();
      data->set_continuation(
        [p = std::move(p)] (unique_future<U>&& in) mutable {
          in.data->set_continuation(
            [p = std::move(p), in = std::move(in)] (U&& u) mutable {
              p.return_value(std::move(u));
            }
          );
        }
      );
      return std::move(g);
    } else
      return std::move(*this);
  }


  std::optional<T> try_get() {
    if (await_ready()) {
      std::optional<T> v;

      // Since the data is ready, this will run immediately in this thread.
      data->set_continuation([&] (T&& t) { v = std::move(t); });

      assert(v);
      return std::move(*v);
    }

    return {};
  }

  template <typename BoostBlocker>
  T get(BoostBlocker&& bb) {
    std::binary_semaphore sem;

    std::optional<T> v;

    // TODO: Should this be using `then`?
    data->set_continuation(
      [&] (T&& t) {
        SYNCLOG("retrieving value");
        v = std::move(t);
        sem.release();
      }
    );

    while (!sem.try_acquire())
      bb.boost_block();

    assert(v);
    return std::move(*v);
  }

  bool await_ready() {
    if (data)
      return data->ready();
    return false;
  }

  template <typename Promise>
  void await_suspend(std::experimental::coroutine_handle<Promise> c) {
    data->set_on_completed([c = std::move(c)] () mutable { c(); });
  }

  auto await_resume() {
    assert(await_ready());
    return *try_get();
  }
};

template <typename Submitter, typename Invocable, typename... Args>
auto async(Submitter&& s, Invocable&& f, Args&&... args) {
  unique_promise<decltype(std::declval<Invocable>()(std::declval<Args>()...))> p;
  // We have to retrieve the future before we move the promise into the
  // lambda below.
  auto g = p.get_return_object();
  s.submit(
    [f = std::forward<Invocable>(f),
     // We have to either copy or move args into the lambda, as the lambda may
     // outlive the current scope.
     args = std::tuple(std::forward<Args>(args)...),
     p = std::move(p)]
    () mutable {
      p.return_value(std::apply(std::move(f), std::move(args)));
    }
  );
  return g.unwrap();
}

template <typename T>
unique_future<std::vector<T>> when_all(std::vector<unique_future<T>>& fs) {
  if (fs.empty()) return std::vector<T>{};

  // FIXME: Only works for default constructible types.
  auto results = std::make_shared<std::vector<T>>(fs.size());

  auto av = std::make_shared<asynchronous_value<std::vector<T>>>();
  unique_future<std::vector<T>> g(av);

  auto gate = std::make_shared<std::atomic<std::uint64_t>>(0);

  for (std::uint64_t i = 0; i < fs.size(); ++i) {
    fs[i].submit(
      [av, i, results, gate] (T&& t) mutable {
        SYNCLOG("assigning when_all result[" << i << "]");
        (*results)[i] = std::move(t);
        std::uint64_t expected = gate->load(std::memory_order_acquire);
        while (!gate->compare_exchange_strong(expected, expected + 1,
                                              std::memory_order_acq_rel))
          ;
        if (expected + 1 == results->size()) {
          SYNCLOG("setting when_all promise");
          av->set_value(std::move(*results));
        }
      }
    );
  }

  return g;
}

namespace std {

template <typename InputIt, typename T, typename BinaryOp>
T reduce(InputIt first, InputIt last, T init, BinaryOp op)
{
  for (; first != last; ++first) {
    init = op(std::move(init), *first);
  }
  return init;
}

template <typename InputIt, typename T>
T reduce(InputIt first, InputIt last, T init)
{
  for (; first != last; ++first)
    init = std::move(init) + *first;
  return init;
}

template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
OutputIt exclusive_scan(InputIt first, InputIt last,
                              OutputIt result, T init, BinaryOp op)
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

template <typename InputIt, typename OutputIt, typename T>
OutputIt exclusive_scan(InputIt first, InputIt last,
                        OutputIt result, T init)
{
  return exclusive_scan(first, last, result, init, std::plus{});
}

template <typename InputIt, typename OutputIt, typename BinaryOp,
          typename T>
OutputIt inclusive_scan(InputIt first, InputIt last,
                        OutputIt result, BinaryOp op, T init)
{
  for (; first != last; ++first, (void) ++result) {
    init = op(init, *first);
    *result = init;
  }
  return result;
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
OutputIt inclusive_scan(InputIt first, InputIt last,
                        OutputIt result, BinaryOp op)
{
  if (first != last) {
    typename std::iterator_traits<InputIt>::value_type init = *first;
    *result++ = init;
    if (++first != last)
      return inclusive_scan(first, last, result, op, init);
  }

  return result;
}

} // namespace std

// TODO: Test this for size = 0, size = 1, size = 2, etc.
template <typename ExecutionPolicy, typename InputIt, typename BinaryOp, typename T>
unique_future<T>
async_reduce(ExecutionPolicy&& exec, InputIt first, InputIt last, T init, BinaryOp op)
{
  std::uint64_t const elements = std::distance(first, last);
  std::uint64_t const chunks   = (1 + ((elements - 1) / exec.chunk_size())); // Round up.

  std::vector<unique_future<T>> sweep;
  assert(chunks);
  sweep.reserve(chunks);

  for (std::uint64_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      [=, &exec] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("sweep chunk(" << this_begin << ", " << this_end << ")");
        return std::reduce(first + this_begin, first + this_end, T{}, op);
      }
    ));

  auto sums = co_await when_all(sweep);

  for (std::uint64_t chunk = 0; chunk < chunks; ++chunk)
    ALGOLOG("sums[" << chunk << "](" << sums[chunk] << ")");

  // We add in init here.
  co_return std::reduce(sums.begin(), sums.end(), init, op);
}

template <typename ExecutionPolicy, typename InputIt>
unique_future<typename std::iterator_traits<InputIt>::value_type>
async_max_element_value(ExecutionPolicy&& exec, InputIt first, InputIt last)
{
  using T = typename std::iterator_traits<InputIt>::value_type;
  if (first == last) return T{};
  auto tmp = *first++;
  return async_reduce(std::forward<ExecutionPolicy>(exec), first, last, std::move(tmp),
                      [] (auto l, auto r) { return std::max(l, r); });
}

struct unused {};

// TODO: Test this for size = 0, size = 1, size = 2, etc.
template <typename ExecutionPolicy, typename InputIt, typename BinaryOp>
unique_future<unused>
async_for_each(ExecutionPolicy&& exec, InputIt first, InputIt last, BinaryOp op)
{
  std::uint64_t const elements = std::distance(first, last);
  std::uint64_t const chunks   = (1 + ((elements - 1) / exec.chunk_size())); // Round up.

  std::vector<unique_future<unused>> sweep;
  assert(chunks);
  sweep.reserve(chunks);

  for (std::uint64_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      [=, &exec] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("sweep chunk(" << this_begin << ", " << this_end << ")");
        std::for_each(first + this_begin, first + this_end, op);
        return unused{};
      }
    ));

  co_await when_all(sweep);

  co_return unused{};
}

// FIXME: Lift this with async_for_each
// TODO: Test this for size = 0, size = 1, size = 2, etc.
template <typename ExecutionPolicy, typename InputIt, typename BinaryOp>
unique_future<unused>
async_for_loop(ExecutionPolicy&& exec, InputIt first, InputIt last, BinaryOp op)
{
  std::uint64_t const elements = std::distance(first, last);
  std::uint64_t const chunks   = (1 + ((elements - 1) / exec.chunk_size())); // Round up.

  std::vector<unique_future<unused>> sweep;
  assert(chunks);
  sweep.reserve(chunks);

  for (std::uint64_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      [=, &exec] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("sweep chunk(" << this_begin << ", " << this_end << ")");
        for (std::uint64_t i = this_begin; i < this_end; ++i)
          op(i, first[i]);
        return unused{};
      }
    ));

  co_await when_all(sweep);

  co_return unused{};
}

// FIXME: Lift this with async_for_each
// TODO: Test this for size = 0, size = 1, size = 2, etc.
template <typename ExecutionPolicy, typename InputIt, typename OutputIt, typename BinaryOp>
unique_future<OutputIt>
async_transform(ExecutionPolicy&& exec, InputIt first, InputIt last, OutputIt output, BinaryOp op)
{
  std::uint64_t const elements = std::distance(first, last);
  std::uint64_t const chunks   = (1 + ((elements - 1) / exec.chunk_size())); // Round up.

  std::vector<unique_future<OutputIt>> sweep;
  assert(chunks);
  sweep.reserve(chunks);

  for (std::uint64_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      [=, &exec] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("sweep chunk(" << this_begin << ", " << this_end << ")");
        return std::transform(first + this_begin, first + this_end,
                              output + this_begin, op);
      }
    ));

  co_await when_all(sweep);

  co_return output + elements;
}

template <typename ExecutionPolicy, typename InputIt, typename OutputIt>
unique_future<OutputIt>
async_copy(ExecutionPolicy&& exec, InputIt first, InputIt last, OutputIt output)
{
  return async_transform(std::forward<ExecutionPolicy>(exec), first, last, output,
                         [] (auto&& t) { return std::forward<decltype(t)>(t); });
}

// TODO: Test this for size = 0, size = 1, size = 2, etc.
// TODO: Test this for small chunk sizes.
template <typename ExecutionPolicy, typename InputIt, typename OutputIt,
          typename BinaryOp, typename T>
unique_future<OutputIt>
async_exclusive_scan(ExecutionPolicy&& exec,
                     InputIt first, InputIt last, OutputIt output,
                     T init, BinaryOp op)
{
  std::size_t const elements = std::distance(first, last);
  std::size_t const chunks   = (1 + ((elements - 1) / exec.chunk_size())); // Round up.

  std::vector<unique_future<T>> sweep;
  sweep.reserve(chunks);

  // Upsweep.
  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      [=, &exec] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("upsweep chunk(" << this_begin << ", " << this_end << ")");
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

  auto sums = co_await when_all(sweep);

  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    ALGOLOG("pre top level scan sums[" << chunk << "](" << sums[chunk] << ")");

  // We add in init here.
  std::inclusive_scan(sums.begin(), sums.end(), sums.begin(), op, init);

  for (std::size_t chunk = 0; chunk < chunks; ++chunk)
    ALGOLOG("post top level scan sums[" << chunk << "](" << sums[chunk] << ")");

  sweep.clear();

  // Downsweep.
  for (std::size_t chunk = 1; chunk < chunks; ++chunk)
    sweep.emplace_back(async(exec.submitter(),
      // Taking sums by reference is fine here; we're going to co_await on the
      // futures, which will keep the current scope alive for them.
      [=, &exec, &sums] {
        auto const this_begin = chunk * exec.chunk_size();
        auto const this_end   = std::min(elements, (chunk + 1) * exec.chunk_size());
        ALGOLOG("downsweep chunk(" << this_begin << ", " << this_end << ")");
        std::for_each(output + this_begin, output + this_end,
                      [=, &sums] (auto& t) {
                        ALGOLOG("downsweep for_each t(" << t
                                << ") sums[" << chunk - 1
                                << "](" << sums[chunk - 1] << ")");
                        t = op(std::move(t), sums[chunk - 1]);
                      });
        return T(*(output + this_end - 1));
      }
    ));

  co_await when_all(sweep);

  co_return output + elements;
}

template <typename ExecutionPolicy, typename InputIt, typename OutputIt>
unique_future<std::uint64_t>
async_radix_sort_split(ExecutionPolicy&& exec,
                       InputIt first, InputIt last, OutputIt output,
                       std::uint64_t bit)
{
  std::uint64_t const elements = std::distance(first, last);

  // TODO: This probably needs less storage.
  std::vector<std::uint64_t> e(elements);

  // Count 0s.
  co_await async_transform(exec,
                           first, last, e.begin(),
                           [=] (auto t) { return !(t & (1 << bit)); });

  for (std::uint64_t i = 0; i < e.size(); ++i)
    ALGOLOG("pre scan zeros_found[" << i << "](" << e[i] << ")");

  // Count the last one if it's set, as we won't get it on the scan.
  std::uint64_t total_falses = e.back();

  co_await async_exclusive_scan(exec,
                                e.begin(), e.end(), e.begin(),
                                std::uint64_t(0), std::plus{});

  total_falses += e.back();

  for (std::uint64_t i = 0; i < e.size(); ++i)
    ALGOLOG("post scan zeros_found[" << i << "](" << e[i] << ")");

  ALGOLOG("total_falses(" << total_falses << ")");

  // Compute destination indices.
  co_await async_for_loop(exec,
                          e.begin(), e.end(),
                          [=] (std::uint64_t i, auto& x) {
                            if (first[i] & (1 << bit)) {
                              ALGOLOG("adjusting destination first["
                                      << i << "](" << first[i] << ") from e["
                                      << i << "](" << x << ") to e["
                                      << i << "](" << (i - x + total_falses)
                                      << ")");
                              assert(i - x + total_falses < elements);
                              x = i - x + total_falses;
                            }
                          });

  for (std::uint64_t i = 0; i < e.size(); ++i) {
    ALGOLOG("destinations[" << i << "](" << e[i] << ")");
  }

  // Scatter.
  co_await async_for_loop(exec,
                          first, last,
                          [=, &e] (std::uint64_t i, auto& x) {
                            ALGOLOG("scatter first[" << i << "](" << first[i]
                                    << ") to e[" << i << "](" << e[i] << ")");
                            output[e[i]] = x;
                          });

  for (std::uint64_t i = 0; i < e.size(); ++i) {
    ALGOLOG("result[" << i << "](" << output[i] << ")");
  }

  co_return total_falses;
}

template <typename ExecutionPolicy, typename InputIt>
unique_future<std::uint64_t>
async_radix_sort(ExecutionPolicy&& exec, InputIt first, InputIt last)
{
  using T = typename std::iterator_traits<InputIt>::value_type;

  constexpr std::uint64_t element_bits = sizeof(T) * CHAR_BIT;

  std::uint64_t const elements = std::distance(first, last);

  ALGOLOG("elements(" << elements << ")");

  if (0 == elements) co_return 0;

  // Figure out how many passes we need to do by find the largest element in
  // the input and determining its most significant set bit.
  std::uint64_t const min_leading_zeros =
    // `__builtin_clz` has UB if the input is 0, thus the | 1.
    __builtin_clzll(
      co_await async_max_element_value(exec, first, last) | 1
    );

  assert(min_leading_zeros <= element_bits);
  std::uint64_t const max_set_bit = element_bits - min_leading_zeros;

  ALGOLOG("element_bits(" << element_bits
          << ") min_leading_zeros(" << min_leading_zeros
          << ") max_set_bit(" << max_set_bit << ")");

  assert(elements);
  std::vector<T> v(elements);

  for (std::uint64_t bit = 0; bit < max_set_bit; ++bit) {
    if (bit % 2 == 0) {
      co_await async_radix_sort_split(exec,
                                      first, last, v.begin(),
                                      bit);
      for (std::uint64_t i = 0; i < elements; ++i)
        ALGOLOG("pass(" << bit << ") of max_pass(" << max_set_bit - 1
                << ") v[" << i << "](" << v[i] << ")");
    }
    else {
      co_await async_radix_sort_split(exec,
                                      v.begin(), v.end(), first,
                                      bit);
      for (std::uint64_t i = 0; i < elements; ++i)
        ALGOLOG("pass(" << bit << ") of max_pass(" << max_set_bit - 1
                << ") first[" << i << "](" << first[i] << ")");
    }
  }

  if (max_set_bit % 2 != 0)
    // Gotta do a copy back.
    co_await async_copy(exec, v.begin(), v.end(), first);

  co_return max_set_bit;
}

template <typename InputIt, typename OutputIt>
std::uint64_t
radix_sort_split(InputIt first, InputIt last, OutputIt output, std::uint64_t bit)
{
  // TODO: This probably needs less storage.
  std::vector<std::uint64_t> e(std::distance(first, last));

  // Count 0s.
  std::transform(first, last, e.begin(),
                 [=] (auto t) { return !(t & (1 << bit)); });

  // Count the last one if it's set, as we won't get it on the scan.
  std::uint64_t total_falses = e.back();

  std::exclusive_scan(e.begin(), e.end(), e.begin(), std::uint64_t(0));

  total_falses += e.back();

  // Compute destination indices.
  for (std::uint64_t i = 0; i < e.size(); ++i)
    if ((first[i] & (1 << bit))) e[i] = i - e[i] + total_falses;

  // Scatter.
  for (std::uint64_t i = 0; i < e.size(); ++i)
    output[e[i]] = first[i];

  return total_falses;
}

template <typename InputIt>
std::uint64_t radix_sort(InputIt first, InputIt last)
{
  using T = typename std::iterator_traits<InputIt>::value_type;

  constexpr std::uint64_t element_bits = sizeof(T) * CHAR_BIT;

  std::uint64_t const elements = std::distance(first, last);

  if (0 == elements) return 0;

  // Figure out how many passes we need to do by find the largest element in
  // the input and determining its most significant set bit.
  std::uint64_t const min_leading_zeros =
    // `__builtin_clz` has UB if the input is 0, thus the | 1.
    __builtin_clzll(*std::max_element(first, last) | 1);

  assert(min_leading_zeros <= element_bits);
  std::uint64_t const max_set_bit = element_bits - min_leading_zeros;

  assert(elements);
  std::vector<T> v(elements);

  for (std::uint64_t bit = 0; bit < max_set_bit; ++bit) {
    if (bit % 2 == 0)
      radix_sort_split(first, last, v.begin(), bit);
    else
      radix_sort_split(v.begin(), v.end(), first, bit);
  }

  if (max_set_bit % 2 != 0)
    // Gotta do a copy back.
    std::copy(v.begin(), v.end(), first);

  return max_set_bit;
}

#if 0
template <std::ranges::random_access_range I, std::ranges::random_access_range O,
          typename T, std::invocable</* ... */> BO>
#endif
template <typename I, typename O,
          typename T, typename BO>
struct exclusive_scanner {
private:
  I  first;
  I  last;
  O  output;
  T  init;
  BO op;

  std::uint64_t const concurrency;
  std::uint64_t const elements;
  std::uint64_t const chunk_size;

  std::vector<T>                      aggregates;
  std::barrier<std::function<void()>> upsweep_gate;
  std::latch                          downsweep_gate;

  void scan_aggregates() {
    // Add in the initial value here.
    std::exclusive_scan(aggregates.begin(), aggregates.end(), aggregates.begin(), std::move(init), op);

    for (std::size_t i = 0; i < concurrency; ++i)
      ALGOLOG("aggregates[" << i << "](" << aggregates[i] << ")");
  }

public:
  exclusive_scanner(std::uint64_t concurrency_, I first_, I last_, O output_, T init_, BO op_)
    : first(std::move(first_))
    , last(std::move(last_))
    , output(std::move(output_))
    , init(std::move(init_))
    , op(std::move(op_))
    , concurrency(concurrency_)
    , elements(std::distance(first, last))
    , chunk_size((elements + concurrency - 1) / concurrency)
    , aggregates(concurrency)
    , upsweep_gate(concurrency, [&] { scan_aggregates(); })
    , downsweep_gate(concurrency)
  {}

  template <typename Submitter>
  unique_future<unused> launch(Submitter&& s) {
    std::vector<unique_future<unused>> workers;
    workers.reserve(concurrency);

    for (std::uint64_t chunk = 0; chunk < concurrency; ++chunk) {
      ALGOLOG("launching chunk(" << chunk << ") of concurrency("
              << concurrency << ")");
      workers.emplace_back(async(s,
        [&, chunk] {
          auto const this_begin = chunk * chunk_size;
          auto const this_end   = std::min(elements, (chunk + 1) * chunk_size);
          process_chunk(chunk,
                        first + this_begin,
                        first + this_end,
                        output + this_begin);
          return unused{};
        }
      ));
    }

    co_await when_all(workers);

    co_return unused{};
  }

  void process_chunk(std::uint64_t chunk, I first, I last, O output) {
    ALGOLOG("executing chunk(" << chunk << ")");
    // Upsweep.
    auto const last_element = *(last - 1);
    //aggregates[chunk] = op(*--std::exclusive_scan(first, last, output, T{}, op), last_element);
    auto tmp = op(*--std::exclusive_scan(first, last, output, T{}, op), last_element);
    ALGOLOG("assigning tmp(" << tmp << ") to aggregates[" << chunk
            << "](" << aggregates[chunk] << ")");
    aggregates[chunk] = tmp;

    // Completion function does the aggregate sum.
    ALGOLOG("chunk(" << chunk << ") arriving at upsweep_gate");
    upsweep_gate.arrive_and_wait();
    ALGOLOG("chunk(" << chunk << ") arrived at upsweep_gate");

    // Downsweep.
//    if (0 != chunk) {
      ALGOLOG("after aggregate aggregates[" << chunk/* - 1*/
              << "](" << aggregates[chunk/* - 1*/] << ")");

      std::for_each(output, output + std::distance(first, last),
                    [&, chunk] (auto& t) {
                      ALGOLOG("downsweep for_each t(" << t
                              << ") aggregates[" << chunk/* - 1*/
                              << "](" << aggregates[chunk/* - 1*/] << ")");
                      t = op(std::move(t), aggregates[chunk/* - 1*/]);
                    });
//    }

    ALGOLOG("chunk(" << chunk << ") arriving at downsweep_gate");
    downsweep_gate.arrive_and_wait();
    ALGOLOG("chunk(" << chunk << ") arrived at downsweep_gate");
  }
};

int main(int argc, char** argv) {
  // TODO: Try with float/double?
  using T = std::uint64_t;

  std::uint64_t threads         = 6;
  std::uint64_t elements        = 2 << 23;
  std::uint64_t task_chunk_size = 2 << 16;

  if (2 <= argc) threads         = std::atoll(argv[1]);
  if (3 <= argc) elements        = std::atoll(argv[2]);
  if (4 <= argc) task_chunk_size = std::atoll(argv[3]);

  double const elements_size_gb   = sizeof(T) * CHAR_BIT * elements / 1e9;
  std::uint64_t const task_chunks = (1 + ((elements - 1) / task_chunk_size));

  std::cout << "threads(" << threads
            << ") elements(" << elements
            << ") elements_size(" << elements_size_gb
            << " [GB]) task_chunk_size(" << task_chunk_size
            << ") task_chunks(" << task_chunks << ")\n";

  std::vector<std::uint64_t> gold(elements);
  std::vector<std::uint64_t> parallel_task(elements);
  std::vector<std::uint64_t> parallel_collective(elements);

  {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<std::uint64_t> dis(0, 4);
    std::generate_n(gold.begin(), elements, [&] { return dis(gen); });
    std::copy(gold.begin(), gold.end(), parallel_task.begin());
    std::copy(gold.begin(), gold.end(), parallel_collective.begin());
  }

  ALGOLOG("initialization complete");

  if (512 >= elements)
    for (std::uint64_t i = 0; i < elements; ++i)
      std::cout << "initial[" << i << "](" << gold[i] << ")\n";

  double time_gold = 0.0;

  {
    auto const start = std::chrono::high_resolution_clock::now();
    std::exclusive_scan(gold.begin(), gold.end(), gold.begin(),
                        std::uint64_t(4), std::plus{});
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time(end - start);
    time_gold = time.count();
  }

  ALGOLOG("gold complete");

  double time_parallel_task = 0.0;

  {
    work_stealing_task_manager tm(threads);

    auto const start = std::chrono::high_resolution_clock::now();
    async_exclusive_scan(execution_policy(tm, task_chunk_size),
                         parallel_task.begin(),
                         parallel_task.end(),
                         parallel_task.begin(),
                         std::uint64_t(4),
                         std::plus{}).get(tm);
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time(end - start);
    time_parallel_task = time.count();
  }

  ALGOLOG("parallel_task complete");

  double time_parallel_collective = 0.0;

  {
    work_stealing_task_manager tm(threads);

    auto const start = std::chrono::high_resolution_clock::now();
    exclusive_scanner es(threads,
                         parallel_collective.begin(),
                         parallel_collective.end(),
                         parallel_collective.begin(),
                         std::uint64_t(4),
                         std::plus{});
    es.launch(tm).get(tm);
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time(end - start);
    time_parallel_collective = time.count();
  }

  ALGOLOG("parallel_collective complete");

  if (!std::equal(gold.begin(), gold.end(), parallel_task.begin()))
    std::cout << "PARALLEL TASK EXCLUSIVE SCAN RESULT FAILED COMPARISON WITH GOLD\n";
  if (!std::equal(gold.begin(), gold.end(), parallel_collective.begin()))
    std::cout << "PARALLEL COLLECTIVE EXCLUSIVE SCAN RESULT FAILED COMPARISON WITH GOLD\n";

  if (512 >= elements)
    for (std::uint64_t i = 0; i < elements; ++i)
      std::cout << "gold[" << i << "](" << gold[i]
                << ") parallel_task[" << i << "](" << parallel_task[i]
                << ") parallel_collective[" << i << "](" << parallel_collective[i]
                << ")\n";

  std::cout << "time_gold(" << time_gold
            << " [sec]) time_parallel_task(" << time_parallel_task
            << " [sec]) time_parallel_collective(" << time_parallel_collective
            << " [sec])\n";
}

