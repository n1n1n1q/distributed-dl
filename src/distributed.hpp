#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <torch/torch.h>
#include "mpi_wrapper/SerializedTensor.hpp"

#define SerializedTensor SerializedTensorCPU_impl

inline void send(boost::mpi::communicator &world, torch::Tensor &tensor, int dest) {
    SerializedTensor tens{tensor};
    world.send(dest, 0, tens);
}

inline void recv(boost::mpi::communicator &world, torch::Tensor &tensor, int source) {
    SerializedTensor received{torch::zeros_like(tensor)};
    world.recv(source, 0, received);
    received.toTensor(tensor);
}

inline void all_reduce(boost::mpi::communicator &world, torch::Tensor &tensor, int dest = 0) {
    int rank = world.rank();
    int size = world.size();
    torch::Tensor result{torch::zeros_like(tensor)};
    if (dest == rank) {
        for (int i = 0; i < size && i != dest; ++i) {
            torch::Tensor received{torch::zeros_like(tensor)};
            recv(world, received, i);
            result += received;
        }

        for (int i = 0; i < size && i != dest; ++i) {
            send(world, result, i);
        }
    } else {
        send(world, tensor, dest);
        recv(world, result, dest);
    }
    tensor = result;
}

#endif
