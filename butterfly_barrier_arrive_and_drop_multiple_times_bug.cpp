#undef __BARRIER_NO_BUTTERFLY

#include <barrier>
#include <iostream>

int main() {
  constexpr std::size_t n = 2;

  std::barrier b(n, [] { std::cout << "phase completed\n"; });

  for (std::size_t i = 0; i < n; ++i)
    b.arrive_and_drop();

  std::cout << "finished arriving and dropping\n";
}

