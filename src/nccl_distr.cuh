#pragma once
#define NCCLDISTR
#ifdef NCCLDISTR

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <nccl.h>

#define CUDACHECK(cmd) do {                         \
    cudaError_t err = cmd;                            \
    if (err != cudaSuccess) {                         \
      printf("Failed: Cuda error %s:%d '%s'\n",       \
          __FILE__,__LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  
  
#define NCCLCHECK(cmd) do {                         \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
      printf("Failed, NCCL error %s:%d '%s'\n",       \
          __FILE__,__LINE__,ncclGetErrorString(res)); \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

inline void send(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int dest) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclSend(data, size, ncclFloat, dest, comm, stream));
}

inline void recv(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int source) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclRecv(data, size, ncclFloat, source, comm, stream));
}

inline void broadcast(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int root) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclBroadcast(data, data, size, ncclFloat, root, comm, stream));
}

inline void reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int root = 0, ncclRedOp_t op = ncclSum) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclReduce(data, data, size, ncclFloat, op, root, comm, stream));
}

inline void all_reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, ncclRedOp_t op = ncclSum) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclAllReduce(data, data, size, ncclFloat, op, comm, stream));
}

#endif // NCCLDISTR