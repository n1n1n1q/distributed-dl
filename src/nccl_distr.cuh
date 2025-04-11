#pragma once
#define NCCLDISTR
#ifdef NCCLDISTR

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <nccl.h>
#include <boost/mpi.hpp>

#define CUDACHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        cudaError_t err = cmd;                                   \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("Failed: Cuda error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

#define NCCLCHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        ncclResult_t res = cmd;                                  \
        if (res != ncclSuccess)                                  \
        {                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

class ProcessGroupNCCL {
  std::unique_ptr<boost::mpi::environment> env_ptr = nullptr;
  std::unique_ptr<boost::mpi::communicator> world_ptr = nullptr;

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t stream;

  std::unique_ptr<torch::Device> device;

  static ncclDataType_t get_nccl_datatype(const torch::Dtype dtype) {
    switch (dtype) {
      case torch::kFloat32: return ncclFloat;
      case torch::kFloat16: return ncclFloat16;
      case torch::kFloat64: return ncclDouble;
      case torch::kInt32: return ncclInt32;
      case torch::kInt64: return ncclInt64;
      case torch::kUInt8: return ncclUint8;
      default:
        throw std::runtime_error("Unsupported dtype for NCCL");
    }
  }

public:
  [[nodiscard]] int rank() const {
    return world_ptr->rank();
  }

  [[nodiscard]] int size() const {
    return world_ptr->size();
  }

  ProcessGroupNCCL(int argc, char **argv) {
    env_ptr = std::make_unique<boost::mpi::environment>(argc, argv);
    world_ptr = std::make_unique<boost::mpi::communicator>();

    if (rank() == 0) {
      ncclGetUniqueId(&id);
    }
    boost::mpi::broadcast(*world_ptr, id.internal, NCCL_UNIQUE_ID_BYTES, 0);


    device = std::make_unique<torch::Device>(torch::kCUDA, rank());
    CUDACHECK(cudaSetDevice(rank()));
    NCCLCHECK(ncclCommInitRank(&comm, size(), id, rank()));
    CUDACHECK(cudaStreamCreate(&stream));
  }

  void send(const torch::Tensor &tensor, int dest) {
    const int64_t num_elements = tensor.numel();
    auto has_grad = tensor.grad().numel() != 0;

    world_ptr->send(dest, 0, has_grad);

    if (has_grad) {
      auto nccl_type = get_nccl_datatype(tensor.grad().scalar_type());
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(
        ncclSend(tensor.mutable_grad().data_ptr(),
          tensor.grad().numel(),
          nccl_type,
          dest, comm, stream));
    }

    NCCLCHECK(ncclSend(
      tensor.data_ptr(),
      num_elements,
      get_nccl_datatype(tensor.scalar_type()),
      dest,
      comm,
      stream
    ));

    if (has_grad)
      NCCLCHECK(ncclGroupEnd());
  }

  void recv(torch::Tensor &tensor, int src) {
    bool has_grad = 0;
    world_ptr->recv(src, 0, has_grad);

    if (has_grad) {
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv(
        tensor.mutable_grad().data_ptr(),
        tensor.grad().numel(),
        get_nccl_datatype(tensor.grad().scalar_type()),
        src,
        comm,
        stream
      ));
    }

    NCCLCHECK(ncclRecv(
      tensor.data_ptr(),
      tensor.numel(),
      get_nccl_datatype(tensor.scalar_type()),
      src,
      comm,
      stream
    ));
    if (has_grad)
      NCCLCHECK(ncclGroupEnd());

    CUDACHECK(cudaStreamSynchronize(stream));
  }

  void broadcast(torch::Tensor &tensor, const int root) {
    auto has_grad = tensor.grad().numel() != 0;
    if (has_grad) {
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(
        ncclBcast(tensor.mutable_grad().data_ptr(), tensor.grad().numel(), get_nccl_datatype(tensor.grad().scalar_type()
        ), root, comm, stream));
    }

    NCCLCHECK(ncclBcast(tensor.data_ptr(), tensor.numel(), get_nccl_datatype(tensor.scalar_type()), root, comm, stream))
    ;

    if (has_grad)
      NCCLCHECK(ncclGroupEnd());
  }

  void reduce(torch::Tensor &tensor, int root = 0,
              ncclRedOp_t op = ncclSum) {
    NCCLCHECK(
      ncclReduce(
        tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        get_nccl_datatype(tensor.scalar_type()),
        op, root, comm, stream));
  }

  void all_reduce(torch::Tensor &tensor, ncclRedOp_t op = ncclSum) {
    NCCLCHECK(
      ncclAllReduce(tensor.data_ptr(),
        tensor.data_ptr(),
        tensor.numel(),
        get_nccl_datatype(tensor.scalar_type()),
        op, comm, stream));
  }

  // void gather(torch::Tensor &send_tensor, torch::Tensor &recv_tensor, int root) {
  //   size_t count = send_tensor.numel();
  //   NCCLCHECK(ncclGroupStart());
  //   if (rank() == root) {
  //     auto *src = send_tensor.data_ptr();
  //     auto *dest = recv_tensor.data_ptr() + root * count;
  //     CUDACHECK(cudaMemcpyAsync(dest, src, count * sizeof(float),
  //       cudaMemcpyDeviceToDevice, stream));
  //     for (int i = 0; i < size(); i++) {
  //       if (i == root)
  //         continue;
  //       float *recv_ptr = recv_tensor.data_ptr<float>() + i * count;
  //       NCCLCHECK(ncclRecv(recv_ptr, count, ncclFloat, i, comm, stream));
  //     }
  //   } else {
  //     NCCLCHECK(ncclSend(send_tensor.data_ptr<float>(), count, ncclFloat, root, comm, stream));
  //   }
  //   NCCLCHECK(ncclGroupEnd());
  //   CUDACHECK(cudaStreamSynchronize(stream));
  // }

  // inline void scatter(torch::Tensor &tensor, int root = 0)

  // {
  //     size_t count = recv_tensor.numel();
  //     NCCLCHECK(ncclGroupStart());
  //     if (rank == root)
  //     {
  //         for (int i = 0; i < nranks; i++)
  //         {
  //             float *src_ptr = send_tensor.data_ptr<float>() + i * count;
  //             if (i == root)
  //             {
  //                 CUDACHECK(cudaMemcpyAsync(recv_tensor.data_ptr<float>(), src_ptr,
  //                                           count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  //             }
  //             else
  //             {
  //                 NCCLCHECK(ncclSend(src_ptr, count, ncclFloat, i, comm, stream));
  //             }
  //         }
  //     }
  //     else
  //     {
  //         NCCLCHECK(ncclRecv(recv_tensor.data_ptr<float>(), count, ncclFloat, root, comm, stream));
  //     }
  //     NCCLCHECK(ncclGroupEnd());
  //     CUDACHECK(cudaStreamSynchronize(stream));
  // }

  void barrier() const {
    float *dummy_recv;
    CUDACHECK(cudaMalloc(&dummy_recv, sizeof(float)));
    float dummy_val = 14.48f;
    CUDACHECK(cudaMemcpy(dummy_recv, &dummy_val, sizeof(float), cudaMemcpyHostToDevice));
    NCCLCHECK(ncclAllReduce(dummy_recv, dummy_recv, 1, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
  }

  ~ProcessGroupNCCL() {
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
  }
};


#endif // NCCLDISTR
