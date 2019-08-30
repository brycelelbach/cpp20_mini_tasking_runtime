#include <cjdb/concepts.hpp>

namespace std { using namespace cjdb; }

template <typename T>
struct concurrent_bounded_queue
{
  void enqueue(std::convertible_to<T> auto&& t) {}

  void enqueue(std::range auto&& r) {}
};

struct range {
  void* begin() { return nullptr; }
  void* end() { return nullptr; }
};

int main() {
  concurrent_bounded_queue<float> c;

  int a = 17;
  c.enqueue(a);
  c.enqueue(17);

  range r{};
  c.enqueue(r);
  c.enqueue(range{});
}

