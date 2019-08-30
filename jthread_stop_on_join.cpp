#include <jthread>

int main()
{
  std::jthread t(
    [&] (std::stop_token s) { while (!s.stop_requested()); }
  );
}

