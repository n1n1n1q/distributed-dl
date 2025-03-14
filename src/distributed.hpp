#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <functional>

#include <torch/torch.h>
#include "mpi_wrapper/SerializedTensor.hpp"

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

inline void reduce(boost::mpi::communicator &world, torch::Tensor &tensor, int dest = 0, std::function<torch::Tensor(torch::Tensor, torch::Tensor)> op = torch::add) {
    int rank = world.rank();
    if (rank == dest) {
        torch::Tensor result = tensor.clone();
        for (int i = 0; i < size; ++i) {
            if (i != dest) {
                torch::Tensor received = torch::zeros_like(tensor);
                recv(world, received, i);
                result = op(result, received);
            }
        }
        tensor = result;
    } else {
        send(world, tensor, dest);
    }
}

inline void broadcast(boost::mpi::communicator &world, torch::Tensor &tensor, int source = 0) {
    SerializedTensor broadcasted{tensor};
    world.broadcast(source, 0, broadcasted);
    broadcasted.toTensor(tensor);
}

#endif
