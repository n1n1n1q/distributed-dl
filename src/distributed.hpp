#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <boost/mpi.hpp>
#include <boost/serialization/split_member.hpp>
#include <torch/torch.h>
#include "serialized_tensor.hpp"

inline void send(mpi::communicator& world, torch::Tensor& tensor, int dest) {
    SerializedTensor tens{tensor};
    boost::mpi::synchronize(world);
    world.send(dest, 0, tens);
}

inline void receive(mpi::communicator& world, torch::Tensor& tensor, int source) {
    boost::mpi::synchronize(world);
    SerializedTensor received{torch::zeros_like(tensor)};
    world.receive(source, 0, received);
    tensor = received.toTensor();
}

inline void all_reduce(mpi::communicator& world, torch::Tensor& tensor, int dest = 0) {
    int rank = world.rank();
    int size = world.size();
    torch::Tensor result{torch::zeros_like(tensor)};
    if (dest == rank) {
        for (int i = 0; i < size && i != dest; ++i) {
            torch::tensor received{torch::zeros_like(tensor)};
            receive(world, received, i);
            result += received;
        }

        for (int i = 0; i < size && i != dest; ++i) {
            send(world, result, i);
        }
    }
    else {
        send(world, tensor, dest);
        receive(world, result, dest);
    }
    tensor = result.toTensor();
}

#endif