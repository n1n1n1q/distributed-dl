#include "serializedtensor.hpp"
#include "distributed.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>

void test_send_recv(ProcessGroupMPI &pg) {
  const int world_size = pg.size();
  const int rank = pg.rank();

  if (world_size < 2) {
    if (rank == 0)
      std::cerr << "Send/Recv test requires at least 2 processes." << std::endl;
    pg.barrier();
    return;
  }

  if (rank == 0) {
    torch::Tensor tensor = torch::ones({2, 2}, torch::kFloat32);
    std::cout << "Rank 0 sending tensor:\n"
        << tensor << std::endl;
    pg.send(tensor, 1);
  } else if (rank == 1) {
    torch::Tensor tensor = torch::zeros({2, 2}, torch::kFloat32);
    pg.recv(tensor, 0);
    std::cout << "Rank 1 received tensor:\n"
        << tensor << std::endl;
  } else {
    std::cout << "Rank " << rank << " didn't recieve or send anything" << std::endl;
  }
  pg.barrier();
}

void test_broadcast(ProcessGroupMPI &pg) {
  const int rank = pg.rank();
  torch::Tensor tensor;
  if (rank == 0) {
    tensor = torch::arange(0, 4, torch::kFloat32).reshape({2, 2});
    std::cout << "Rank 0 broadcasting tensor:\n"
        << tensor << std::endl;
  } else {
    tensor = torch::zeros({2, 2}, torch::kFloat32);
  }

  pg.broadcast(tensor, 0);

  std::cout << "Rank " << rank << " tensor after broadcast:\n"
      << tensor << std::endl;
  pg.barrier();
}

void test_async_send_recv(ProcessGroupMPI &pg) {
  const int world_size = pg.size();
  const int rank = pg.rank();

  if (world_size < 2) {
    if (rank == 0)
      std::cerr << "Async send/recv test requires at least 2 processes." << std::endl;
    pg.barrier();
    return;
  }

  if (rank == 0) {
    torch::Tensor tensor = torch::full({2, 2}, 5, torch::kFloat32);
    std::cout << "Rank 0 asynchronously sending tensor:\n"
        << tensor << std::endl;
    auto request = pg.isend(tensor, 1);
    std::cout << "Rank " << rank << " asynchronously sent tensor and waiting" << std::endl;
    request.wait();
    std::cout << "Rank " << rank << " asynchronously sent tensor and success" << std::endl;
  } else if (rank == 1) {
    std::cout << "Rank 1 asynchronously preparing receiving tensor:" << std::endl;
    torch::Tensor tensor = torch::zeros({2, 2}, torch::kFloat32);
    std::cout << "Rank 1 asynchronously receiving tensor:" << std::endl;
    auto request = pg.irecv(tensor, 0);
    std::cout << "Rank 1 lol:" << std::endl;
    request.wait();
    std::cout << "Rank 1 meow:" << std::endl;
    std::cout << "Rank 1 asynchronously received tensor:\n"
        << tensor << std::endl;
  }
  pg.barrier();
}

void test_default_async(int argc, char **argv) {
  std::cout << "testing async" << std::endl;

  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  auto size = world.size();
  auto rank = world.rank();

  if (rank == 0) {
    std::string msg = "hi";
    auto tensor = torch::rand({2, 2}, torch::kFloat32);
    auto t = SerializedTensorCPU_impl{tensor};
    auto request = world.isend(1, 0, t);
    std::cout << "---\nsent\n" << tensor << "---" << std::endl;
  } else if (rank == 1) {
    auto tensor = torch::zeros({2, 2}, torch::kFloat32);
    std::cout << "----\nbefore:\n" << tensor << "----" << std::endl;
    auto t = SerializedTensorCPU_impl{tensor};
    auto request = world.irecv(0, 0, t);
    request.wait();
    t.toTensor(tensor);

    std::cout << "----\nafter:\n" << tensor << "----" << std::endl;
  }
}

int main(int argc, char **argv) {
  ProcessGroupMPI pg(argc, argv);

  std::cout << "Testing send/recv functions" << std::endl;

  test_send_recv(pg);
  test_broadcast(pg);
  // test_async_send_recv(pg);
  test_default_async(argc, argv);

  return 0;
}
