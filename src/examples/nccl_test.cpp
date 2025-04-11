#include <torch/torch.h>
#include <boost/mpi.hpp>
#include "nccl_distr.cuh"


int main(int argc, char **argv) {
  // boost::mpi::environment env(argc, argv);
  // boost::mpi::communicator world;
  //
  // auto size = world.size();
  // auto rank = world.rank();
  //
  //
  // ncclUniqueId id;
  // ncclComm_t comm;
  // cudaStream_t stream;
  //
  // if (rank == 0) {
  //   ncclGetUniqueId(&id);
  // }
  // broadcast(world, id.internal, NCCL_UNIQUE_ID_BYTES, 0);
  //
  // torch::Device device(torch::kCUDA, rank);
  // CUDACHECK(cudaSetDevice(rank));
  // NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
  // CUDACHECK(cudaStreamCreate(&stream));
  //
  // if (rank == 0) {
  //   // Create a sample tensor on GPU 0
  //   torch::Tensor tensor = torch::randn({3, 4}, device).mul(10).to(torch::kFloat32);
  //   std::cout << "Rank 0 sending tensor:\n" << tensor.cpu() << std::endl;
  //
  //   // Get tensor metadata
  //   void *data_ptr = tensor.data_ptr();
  //   const int64_t num_elements = tensor.numel();
  //   const auto dtype = tensor.scalar_type();
  //
  //   // Send metadata first (shape and dtype)
  //   std::vector<int64_t> shape = tensor.sizes().vec();
  //   world.send(1, 0, shape);
  //   world.send(1, 1, dtype);
  //
  //   // Send actual tensor data via NCCL
  //   NCCLCHECK(ncclSend(
  //     data_ptr,
  //     num_elements,
  //     get_nccl_datatype(dtype),
  //     1, // dest rank
  //     comm,
  //     stream
  //   ));
  // } else if (rank == 1) {
  //   // Receive metadata first
  //   std::vector<int64_t> shape;
  //   torch::Dtype dtype;
  //   world.recv(0, 0, shape);
  //   world.recv(0, 1, dtype);
  //
  //   // Create empty tensor on GPU 1
  //   torch::TensorOptions options = torch::TensorOptions()
  //       .device(device)
  //       .dtype(dtype);
  //   torch::Tensor recv_tensor = torch::empty(shape, options);
  //
  //   // Receive data via NCCL
  //   NCCLCHECK(ncclRecv(
  //     recv_tensor.data_ptr(),
  //     recv_tensor.numel(),
  //     get_nccl_datatype(dtype),
  //     0, // src rank
  //     comm,
  //     stream
  //   ));
  //
  //   // Synchronize and print
  //   CUDACHECK(cudaStreamSynchronize(stream));
  //   std::cout << "Rank 1 received tensor:\n" << recv_tensor.cpu() << std::endl;
  // }

  ProcessGroupNCCL pg(argc, argv);

  auto rank = pg.rank();
  auto size = pg.size();

  auto t = torch::rand({2, 2}, torch::kCUDA);
  pg.broadcast(t, 0);
  if (rank == 0) {
    std::cout << "after bcast " << ' ' << rank << " \n" << t.cpu() << std::endl;
  } else if (rank == 1) {
    std::cout << "after bcast " << ' ' << rank << " \n" << t.cpu() << std::endl;
  }
}
