// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

#include <vector>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <iostream>

#include <climits>

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

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result)
{
  return inclusive_scan(first, last, result, std::plus{});
}

} // namespace std

template <typename InputIt, typename OutputIt>
OutputIt
radix_sort_split(InputIt first, InputIt last, OutputIt output, std::uint64_t bit)
{
  std::vector<std::uint64_t> e(std::distance(first, last));

  // Count 0s.
  std::transform(first, last, e.begin(),
                 [=] (auto t) { return !(t & (1 << bit)); });

  // Count the last one if it's set, as we won't get it on the scan.
  std::uint64_t total_falses = e.back();

  std::exclusive_scan(e.begin(), e.end(), e.begin(), std::uint64_t(0));

  total_falses += e.back();

  // Compute destination indices.
  for (std::uint64_t i = 0; i < e.size(); ++i) {
    if ((first[i] & (1 << bit))) e[i] = i - e[i] + total_falses;
  }

  // Scatter.
  for (std::uint64_t i = 0; i < e.size(); ++i)
    output[e[i]] = first[i];

  return output + e.size();
}

int main() {
  constexpr std::uint64_t element_bits = sizeof(std::uint64_t) * CHAR_BIT;

  std::vector<std::uint64_t> u = {
    0b100, 0b111, 0b010, 0b110, 0b011, 0b101, 0b001, 0b000
  };

  for (std::uint64_t i = 0; i < u.size(); ++i)
    std::cout << std::bitset<element_bits>(u[i]) << " " << u[i] << "\n";
  std::cout << "\n";

  std::vector<std::uint64_t> v(u.size());

  std::uint64_t bit = 0;
  do {
    radix_sort_split(u.begin(), u.end(), v.begin(), bit++);

    for (std::uint64_t i = 0; i < v.size(); ++i)
      std::cout << std::bitset<element_bits>(v[i]) << " " << v[i] << "\n";
    std::cout << "\n";

    std::swap(u, v);
  } while (bit < element_bits && !std::is_sorted(u.begin(), u.end()));
}

