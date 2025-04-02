// #include <nccl.h>
#include <torch/torch.h>

#include "distributed.hpp"

int main(int argc, char **argv) {
  auto pg = ProcessGroupMPI(argc, argv);

  auto numranks = pg.size();
  auto rank = pg.rank();

  if (rank == 0) {
    auto t = torch::randn({2, 2});
    std::cout << "sending:\n" << t << std::endl;
    pg.send(t, 1);
  } else if (rank == 1) {
    auto tens = torch::ones({2, 2});

    pg.recv(tens, 0);
    std::cout << "Recieved tensor:\n" << tens << std::endl;
  }
}
