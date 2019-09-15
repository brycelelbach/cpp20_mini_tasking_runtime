#include <cjdb/concepts.hpp>

namespace std { using namespace cjdb; }

struct A {};

void foo0(std::invocable<A&&> auto&& t) {}
void foo1(std::invocable<A&&> auto&& t) {}
void foo2(std::invocable auto&& t) {}

void bar0(A&&) {}
void bar1(A) {}
void bar2() {}

int main() {
  foo0(bar0);
  foo0(bar1);
  //foo0(bar2);

  foo1(bar0);
  foo1(bar1);
  //foo1(bar2);

  //foo2(bar0);
  //foo2(bar1);
  foo2(bar2);
}

