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

inline void broadcast(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int source) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    for (int i = 0; i < comm.nRanks; ++i) {
        if (i == root)
        {
            NCCLCHECK(ncclSend(data, size, ncclFloat, i, comm, stream));
        if (i != source)
        {
            NCCLCHECK(ncclRecv(data, size, ncclFloat, i, comm, stream));
        }
    }
}

inline void reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int dest = 0; F op = ncclSum) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclReduce(data, data, size, ncclFloat, op, dest, comm, stream));
}

inline void all_reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, F op = ncclSum) {
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclAllReduce(data, data, size, ncclFloat, op, comm, stream));
}

#endif // NCCLDISTR