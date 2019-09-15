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

  std::vector<T> aggregates;

  void scan_aggregates() {
    // Add in the initial value here.
    std::inclusive_scan(aggregates.begin(), aggregates.end(), aggregates.begin(), op, std::move(init));

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
  {}

  void launch() {
    // Upsweep.
    for (std::uint64_t chunk = 0; chunk < concurrency; ++chunk) {
      auto const this_begin = chunk * chunk_size;
      auto const this_end   = std::min(elements, (chunk + 1) * chunk_size);
      process_chunk_upsweep(chunk,
                            first + this_begin,
                            first + this_end,
                            output + this_begin);
    }

    // Scan aggregates.
    scan_aggregates();

    // Downsweep.
    for (std::uint64_t chunk = 0; chunk < concurrency; ++chunk) {
      auto const this_begin = chunk * chunk_size;
      auto const this_end   = std::min(elements, (chunk + 1) * chunk_size);
      process_chunk_downsweep(chunk,
                              first + this_begin,
                              first + this_end,
                              output + this_begin);
    }
  }

  void process_chunk_upsweep(std::uint64_t chunk, I first, I last, O output) {
    ALGOLOG("executing chunk(" << chunk << ")");
    // Upsweep.
    auto const last_element = *(last - 1);
    auto const it = --std::inclusive_scan(first, last - 1, output + 1, op);
    ALGOLOG("chunk(" << chunk << ") *it(" << *it << ")");
    aggregates[chunk] = op(*it, last_element);
//    aggregates[chunk] = op(*--std::inclusive_scan(first, last - 1, output + 1, op), last_element);
//    aggregates[chunk] = op(*--std::exclusive_scan(first, last, output, T{}, op), last_element);
//    auto tmp = op(*--std::exclusive_scan(first, last, output, T{}, op), last_element);
//    ALGOLOG("assigning tmp(" << tmp << ") to aggregates[" << chunk
//            << "](" << aggregates[chunk] << ")");
//    aggregates[chunk] = tmp;
  }

  void process_chunk_downsweep(std::uint64_t chunk, I first, I last, O output) {
    // Downsweep.
    if (0 != chunk) {
      ALGOLOG("after aggregate aggregates[" << chunk - 1
              << "](" << aggregates[chunk - 1] << ")");

      *first = aggregates[chunk - 1];
      std::for_each(output + 1, output + std::distance(first, last),
                    [&, chunk] (auto& t) {
                      ALGOLOG("downsweep for_each t(" << t
                              << ") aggregates[" << chunk - 1
                              << "](" << aggregates[chunk - 1] << ")");
                      t = op(std::move(t), aggregates[chunk - 1]);
                    });
    }
  }
};

int main(int argc, char** argv) {
  using T = std::uint64_t;

  std::uint64_t threads   = 6;
  std::uint64_t elements  = 2 << 23;

  if (2 <= argc) threads  = std::atoll(argv[1]);
  if (3 <= argc) elements = std::atoll(argv[2]);

  double const elements_size_gb = sizeof(T) * CHAR_BIT * elements / 1e9;

  std::cout << "threads(" << threads
            << ") elements(" << elements
            << ") elements_size(" << elements_size_gb
            << " [GB])\n";

  std::vector<std::uint64_t> gold(elements);
  std::vector<std::uint64_t> work_efficient(elements);

  {
    std::mt19937 gen(1337);
    std::uniform_int_distribution<std::uint64_t> dis(0, 4);
    std::generate_n(gold.begin(), elements, [&] { return dis(gen); });
    std::copy(gold.begin(), gold.end(), work_efficient.begin());
  }

  ALGOLOG("initialization complete");

  if (512 >= elements)
    for (std::uint64_t i = 0; i < elements; ++i)
      std::cout << "initial[" << i << "](" << gold[i] << ")\n";

  double time_gold = 0.0;

  {
    auto const start = std::chrono::high_resolution_clock::now();
    std::exclusive_scan(gold.begin(), gold.end(), gold.begin(),
                        std::uint64_t(0), std::plus{});
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time(end - start);
    time_gold = time.count();
  }

  ALGOLOG("gold complete");

  double time_work_efficient = 0.0;

  {
    auto const start = std::chrono::high_resolution_clock::now();
    exclusive_scanner es(threads,
                         work_efficient.begin(),
                         work_efficient.end(),
                         work_efficient.begin(),
                         std::uint64_t(0),
                         std::plus{});
    es.launch();
    auto const end   = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time(end - start);
    time_work_efficient = time.count();
  }

  ALGOLOG("work_efficient complete");

  if (!std::equal(gold.begin(), gold.end(), work_efficient.begin()))
    std::cout << "WORK EFFICIENT EXCLUSIVE SCAN RESULT FAILED COMPARISON WITH GOLD\n";

  if (512 >= elements)
    for (std::uint64_t i = 0; i < elements; ++i)
      std::cout << "gold[" << i << "](" << gold[i]
                << ") work_efficient[" << i << "](" << work_efficient[i]
                << ")\n";

  std::cout << "time_gold(" << time_gold
            << " [sec]) time_work_efficient(" << time_work_efficient
            << " [sec])\n";
}

