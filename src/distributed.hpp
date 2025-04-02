#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <torch/torch.h>
#include "serializedtensor.hpp"

#define SerializedTensor SerializedTensorCPU_impl

inline void barrier(mpi::communicator world) {
    world.barrier();
}

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

template<typename F = decltype(default_add)>
inline void all_reduce(boost::mpi::communicator &world, torch::Tensor &tensor, int dest = 0, F op = default_add) {
    int rank = world.rank();
    int size = world.size();
    torch::Tensor result{torch::zeros_like(tensor)};
    if (dest == rank) {
        for (int i = 0; i < size; ++i) {
            if (i != dest) {
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

template<typename F = decltype(default_add)>
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

inline void scatter(boost::mpi::communicator& world, torch::Tensor& tensor, int root = 0) {
    int rank = world.rank();
    if (rank == root) {
        std::vector<torch::Tensor> chunks = tensor.tensor_split(world.size(), 0);
        for (int i = 0; i < world.size(); ++i) {
            if (i == root) {
                tensor = chunks[i].clone();
            } else {
                send(world, chunks[i], i);
            }
        }
    } else {
        recv(world, tensor, root);
    }
}

inline void gather(boost::mpi::communicator& world, torch::Tensor& tensor, 
                    std::vector<torch::Tensor>& gathered_tensors, int root = 0) {
    if (world.rank() == root) {
        gathered_tensors.clear();
        gathered_tensors.reserve(world.size());
        gathered_tensors.push_back(tensor);

        for (int src = 0; src < world.size(); ++src) {
            if (src == root) continue;
            torch::Tensor temp = torch::zeros_like(tensor);
            recv(world, temp, src);
            gathered_tensors.push_back(temp);
        }
    } else {
        send(world, tensor, root);
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
