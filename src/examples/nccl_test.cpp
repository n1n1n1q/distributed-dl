#include <torch/torch.h>
#include <boost/mpi.hpp>
#include "nccl_distr.cuh"

int main(int argc, char **argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  auto size = world.size();
  auto rank = world.rank();


  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t stream;

  if (rank == 0) {
    ncclGetUniqueId(&id);
  }
  broadcast(world, id.internal, NCCL_UNIQUE_ID_BYTES, 0);

  CUDACHECK(cudaSetDevice(rank));
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
  CUDACHECK(cudaStreamCreate(&stream));

  auto t = torch::rand({2,}).cuda();
  if (rank == 0) {
    std::cout << "sending tensor\n" << t << std::endl;
    send(comm, stream, t, 1);
  } else if (rank == 1) {
    auto meow = torch::zeros({2, 2}).cuda();
    recv(comm, stream, meow, 0);
    std::cout << "recv tensor\n " << meow << std::endl;
  }
}
