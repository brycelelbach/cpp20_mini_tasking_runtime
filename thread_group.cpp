#include <jthread>
#include <stop_token>
#include <iostream>
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

  std::vector<std::jthread> members;
};

int main() {
  std::atomic<int> count(0);

  {
    thread_group tg(8, [&] { ++count; });
  }

  // TODO: `format`.
  std::cout << count << std::endl;
}
