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


inline void scatter(mpi::communicator& world, torch::Tensor& tensor, int root = 0) {
    int rank = world.rank();
    if (rank == root) {

        std::vector<torch::Tensor> chunks = tensor.tensor_split(world.size(), 0);
        for (int i = 0; i < world.size(); ++i) {
            if (i == root) {
                tensor = chunks[i];
            } else {
                send(world, chunks[i], i);
            }
        }
    } else {
        receive(world, tensor, root);
    }
}

inline void gather(mpi::communicator& world, std::vector<torch::Tensor>& tensor, int root = 0) {
    int rank = world.rank();
    if (rank == root) {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(world.size());

        for (int src = 0; src < world.size(); ++src) {
            if (src == root) continue;
            torch::Tensor temp = torch::zeros_like(tensor);
            receive(world, temp, src);
            tensors.push_back(temp);
        }

    } else {
        send(world, tensor, root);
    }
}

inline boost::mpi::request isend(mpi::communicator& world, torch::Tensor& tensor, int dest) {
    SerializedTensor tens{tensor};
    boost::mpi::synchronize(world);
    return world.isend(dest, 0, tens);
}

inline void ireceive(mpi::communicator& world, torch::Tensor& tensor, int source) {
    boost::mpi::synchronize(world);
    SerializedTensor received{torch::zeros_like(tensor)};
    world.irecv(source, 0, received);
    tensor = received.toTensor(); need to do this after receiving
}

inline void igather(mpi::communicator& world, std::vector<torch::Tensor>& tensor, int root = 0) {
    int rank = world.rank();
    if (rank == root) {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(world.size());

        for (int src = 0; src < world.size(); ++src) {
            if (src == root) continue;
            torch::Tensor temp = torch::zeros_like(tensor);
            ireceive(world, temp, src);
            tensors.push_back(temp);
        }

    } else {
        isend(world, tensor, root);
    }

    mpi::wait_all(requests.begin(), requests.end());
}

inline void iscatter(mpi::communicator& world, torch::Tensor& tensor, int root = 0) {
    int rank = world.rank();
    if (rank == root) {

        std::vector<torch::Tensor> chunks = tensor.tensor_split(world.size(), 0);
        for (int i = 0; i < world.size(); ++i) {
            if (i == root) {
                tensor = chunks[i];
            } else {
                isend(world, chunks[i], i);
            }
        }
    } else {
        ireceive(world, tensor, root);
    }

    mpi::wait_all(requests.begin(), requests.end());
}

#endif