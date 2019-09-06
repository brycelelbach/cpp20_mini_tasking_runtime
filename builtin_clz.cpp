#include <iostream>
#include <bitset>
#include <vector>

int main() {
  std::vector<std::uint64_t> u = {
    0b0000000,
    0b0000001,
    0b0000010,
    0b0000100,
    0b0001000,
    0b0010000,
    0b0100000,
    0b1000000
  };

  for (std::uint64_t i = 0; i < u.size(); ++i) {
    std::cout << std::bitset<64>(u[i]) << " " << u[i] << " " << __builtin_clzll(u[i] | 1) << "\n";
  }
}

