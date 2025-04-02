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
        } 
        if (i != source)
        {
            NCCLCHECK(ncclRecv(data, size, ncclFloat, i, comm, stream));
        }
    }
}

inline void gather(ncclComm_t comm, cudaStream_t stream, torch::Tensor &input, torch::Tensor &output, int root) {
    int my_rank, n_ranks;
    NCCLCHECK(ncclCommUserRank(comm, &my_rank));
    NCCLCHECK(ncclCommCount(comm, &n_ranks));

    auto input_data = input.data_ptr<float>();
    auto numel = input.numel();

    if (my_rank == root) {
        auto output_data = output.data_ptr<float>();

        if (output.numel() < numel * n_ranks) {
            printf("Output tensor too small: requires %ld elements, has %ld\n", numel * n_ranks, output.numel());
            exit(EXIT_FAILURE);
        }

        CUDACHECK(cudaMemcpyAsync(
            output_data + root * numel,
            input_data,
            numel * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream));

        for (int r = 0; r < n_ranks; ++r) {
            if (r != root) {
                recv(comm, stream, output_data + r * numel, r);
            }
        }
    }
    else {
        send(comm, stream, input, root);
    }
}

inline void scatter(ncclComm_t comm, cudaStream_t stream, torch::Tensor &input,
                    torch::Tensor &output, int root) {
    int my_rank, n_ranks;
    NCCLCHECK(ncclCommUserRank(comm, &my_rank));
    NCCLCHECK(ncclCommCount(comm, &n_ranks));

    const size_t numel = output.numel();

    if (my_rank == root) {
        if (input.numel() < numel * n_ranks) {
            printf("Input tensor too small: requires %ld elements, has %ld\n",
                   numel * n_ranks, input.numel());
            exit(EXIT_FAILURE);
        }

        for (int r = 0; r < n_ranks; ++r) {
            auto chunk = input.slice(0, r * numel, (r + 1) * numel);
            if (r == root) {
                CUDACHECK(cudaMemcpyAsync(
                    output.data_ptr<float>(),
                    chunk.data_ptr<float>(),
                    numel * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
            else {
                send(comm, stream, chunk, r);
            }
        }
    }
    else {
        recv(comm, stream, output, root);
    }
}

#endif // NCCLDISTR