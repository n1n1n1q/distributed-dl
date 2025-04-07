#pragma once
#define NCCLDISTR
#ifdef NCCLDISTR

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <nccl.h>

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

inline void send(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int dest)
{
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclSend(data, size, ncclFloat, dest, comm, stream));
}

inline void recv(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int source)
{
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclRecv(data, size, ncclFloat, source, comm, stream));
}

inline void broadcast(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int root)
{
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclBroadcast(data, data, size, ncclFloat, root, comm, stream));
}

inline void reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int root = 0, ncclRedOp_t op = ncclSum)
{
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclReduce(data, data, size, ncclFloat, op, root, comm, stream));
}

inline void all_reduce(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, ncclRedOp_t op = ncclSum)
{
    auto data = tensor.data_ptr<float>();
    auto size = tensor.numel();
    NCCLCHECK(ncclAllReduce(data, data, size, ncclFloat, op, comm, stream));
}

inline void gather(ncclComm_t comm, cudaStream_t stream,
                   torch::Tensor &send_tensor, torch::Tensor &recv_tensor,
                   int root, int rank, int nranks)
{
    size_t count = send_tensor.numel();
    NCCLCHECK(ncclGroupStart());
    if (rank == root)
    {
        float *src = send_tensor.data_ptr<float>();
        float *dest = recv_tensor.data_ptr<float>() + root * count;
        CUDACHECK(cudaMemcpyAsync(dest, src, count * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
        for (int i = 0; i < nranks; i++)
        {
            if (i == root)
                continue;
            float *recv_ptr = recv_tensor.data_ptr<float>() + i * count;
            NCCLCHECK(ncclRecv(recv_ptr, count, ncclFloat, i, comm, stream));
        }
    }
    else
    {
        NCCLCHECK(ncclSend(send_tensor.data_ptr<float>(), count, ncclFloat, root, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));
}

inline void scatter(ncclComm_t comm, cudaStream_t stream, torch::Tensor &tensor, int root = 0)
{
    size_t count = recv_tensor.numel();
    NCCLCHECK(ncclGroupStart());
    if (rank == root)
    {
        for (int i = 0; i < nranks; i++)
        {
            float *src_ptr = send_tensor.data_ptr<float>() + i * count;
            if (i == root)
            {
                CUDACHECK(cudaMemcpyAsync(recv_tensor.data_ptr<float>(), src_ptr,
                                          count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            }
            else
            {
                NCCLCHECK(ncclSend(src_ptr, count, ncclFloat, i, comm, stream));
            }
        }
    }
    else
    {
        NCCLCHECK(ncclRecv(recv_tensor.data_ptr<float>(), count, ncclFloat, root, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));
}

inline void barrier(ncclComm_t comm, )
{
    float dummy_send = 0.0f, dummy_recv = 0.0f;
    NCCLCHECK(ncclAllReduce(&dummy_send, &dummy_recv, 1, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
}

#endif // NCCLDISTR