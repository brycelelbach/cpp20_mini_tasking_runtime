// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

#include <vector>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <iostream>

#include <climits>
#include <cassert>

namespace std {

template <typename InputIterator, typename T, typename BinaryOperation>
T accumulate(InputIterator first, InputIterator last, T init, BinaryOperation op)
{
  for (; first != last; ++first) {
    init = op(std::move(init), *first);
  }
  return init;
}

template <typename InputIterator, typename T>
T accumulate(InputIterator first, InputIterator last, T init)
{
  for (; first != last; ++first)
    init = std::move(init) + *first;
  return init;
}

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

  // Find the smallest number of leading zeros in the input.
  std::uint64_t min_leading_zeros =
    std::accumulate(u.begin(), u.end(), element_bits,
                    [] (auto l, auto r) {
                      // `__builtin_clz` has UB if the input is 0, thus the | 1.
                      return std::min(l, std::uint64_t(__builtin_clzll(r | 1)));
                    });

  assert(min_leading_zeros <= element_bits);
  std::uint64_t max_set_bit = element_bits - min_leading_zeros;

  std::cout << "element_bits      == " << element_bits << "\n";
  std::cout << "min_leading_zeros == " << min_leading_zeros << "\n";
  std::cout << "max_set_bit       == " << max_set_bit << "\n";
  std::cout << "\n";

  std::vector<std::uint64_t> v(u.size());

  for (std::uint64_t bit = 0; bit < max_set_bit; ++bit) {
    radix_sort_split(u.begin(), u.end(), v.begin(), bit);

    for (std::uint64_t i = 0; i < v.size(); ++i)
      std::cout << std::bitset<element_bits>(v[i]) << " " << v[i] << "\n";
    std::cout << "\n";

    std::swap(u, v);
  }

  if (max_set_bit % 2 == 0)
    // Gotta swap back one more time.
    std::swap(u, v);

  for (std::uint64_t i = 0; i < u.size(); ++i)
    std::cout << std::bitset<element_bits>(u[i]) << " " << u[i] << "\n";
  std::cout << "\n";
}

