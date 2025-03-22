#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <torch/torch.h>
#include "serializedtensor.hpp"

#define SerializedTensor SerializedTensorCPU_impl

static auto default_add = [](const torch::Tensor &a, const torch::Tensor &b) -> torch::Tensor {
    return torch::add(a, b);
};

inline void send(boost::mpi::communicator &world, torch::Tensor &tensor, int dest) {
    SerializedTensor tens{tensor};
    world.send(dest, 0, tens);
}

inline void recv(boost::mpi::communicator &world, torch::Tensor &tensor, int source) {
    SerializedTensor received{torch::zeros_like(tensor)};
    world.recv(source, 0, received);
    received.toTensor(tensor);
}

inline void broadcast(boost::mpi::communicator &world, torch::Tensor &tensor, int source = 0) {
    int rank = world.rank();
    if (rank == source) { 
        int size = world.size();
        for (int i = 0; i < size; ++i) {
            if (i != source) {
                send(world, tensor, i);
            }
        }
    } else {
        recv(world, tensor, source);
    }
}

template <typename F = decltype(default_add)>
inline void all_reduce(boost::mpi::communicator &world, torch::Tensor &tensor, int dest = 0, F op = default_add) {
    int rank = world.rank();
    int size = world.size();
    torch::Tensor result{torch::zeros_like(tensor)};
    if (dest == rank) {
        for (int i = 0; i < size; ++i) {
            if (i != dest){
                torch::Tensor received{torch::zeros_like(tensor)};
                recv(world, received, i);
                result = op(result, received);
            }
        }
        for (int i = 0; i < size; ++i) {
            if (i != dest) {
                send(world, result, i);
            }
        }
    } else {
        send(world, tensor, dest);
        recv(world, result, dest);
    }
    tensor = result;
}

template <typename F = decltype(default_add)>
inline void reduce(boost::mpi::communicator &world, torch::Tensor &tensor, int dest = 0, F op = default_add) {
    int rank = world.rank();
    int size = world.size();
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

inline boost::mpi::request isend(boost::mpi::communicator &world, torch::Tensor &tensor, int dest) {
    SerializedTensor tens{tensor};
    return world.isend(dest, 0, tens);
}

inline boost::mpi::request irecv(boost::mpi::communicator &world, torch::Tensor &tensor, int source) {
    SerializedTensor received{torch::zeros_like(tensor)};
    return world.irecv(source, 0, received);
}

#endif
