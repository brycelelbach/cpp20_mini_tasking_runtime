#include <cjdb/concepts.hpp>

namespace std { using namespace cjdb; }

template <typename T>
void test(T x, std::convertible_to<T> auto&& y) {}

int main() {
  test(int{}, long{});
}

