#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <torch/torch.h>
#include "serializedtensor.hpp"

#define SerializedTensor SerializedTensorCPU_impl


static auto default_add = [](const torch::Tensor &a, const torch::Tensor &b) -> torch::Tensor {
    return torch::add(a, b);
};


class ProcessGroupMPI {
    std::unique_ptr<boost::mpi::environment> env_ptr = nullptr;
    std::unique_ptr<boost::mpi::communicator> world_ptr = nullptr;

public:
    ProcessGroupMPI(int argc, char **argv) {
        env_ptr = std::make_unique<boost::mpi::environment>(argc, argv);
        world_ptr = std::make_unique<boost::mpi::communicator>();
    }

    [[nodiscard]] int size() const {
        return world_ptr->size();
    }

    [[nodiscard]] int rank() const {
        return world_ptr->rank();
    }

    inline void send(torch::Tensor &tensor, int dest) {
        SerializedTensor tens{tensor};
        world_ptr->send(dest, 0, tens);
    }

    inline void recv(torch::Tensor &tensor, int source) {
        SerializedTensor received{torch::zeros_like(tensor)};


        world_ptr->recv(source, 0, received);
        received.toTensor(tensor);
    }

    inline void broadcast(torch::Tensor &tensor, int source = 0) {
        int rank = this->rank();
        if (rank == source) {
            int size = this->size();
            for (int i = 0; i < size; ++i) {
                if (i != source) {
                    send(tensor, i);
                }
            }
        } else {
            recv(tensor, source);
        }
    }

    template<typename F = decltype(default_add)>
    inline void all_reduce(torch::Tensor &tensor, int dest = 0, F op = default_add) {
        int rank = this->rank();
        int size = this->size();
        torch::Tensor result{torch::zeros_like(tensor)};
        if (dest == rank) {
            for (int i = 0; i < size; ++i) {
                if (i != dest) {
                    torch::Tensor received{torch::zeros_like(tensor)};
                    recv(received, i);
                    result = op(result, received);
                }
            }
            for (int i = 0; i < size; ++i) {
                if (i != dest) {
                    send(result, i);
                }
            }
        } else {
            send(tensor, dest);
            recv(result, dest);
        }
        tensor = result;
    }

    template<typename F = decltype(default_add)>
    inline void reduce(torch::Tensor &tensor, int dest = 0, F op = default_add) {
        int rank = this->rank();
        int size = this->size();
        if (rank == dest) {
            torch::Tensor result = tensor.clone();
            for (int i = 0; i < size; ++i) {
                if (i != dest) {
                    torch::Tensor received = torch::zeros_like(tensor);
                    recv(received, i);
                    result = op(result, received);
                }
            }
            tensor = result;
        } else {
            send(tensor, dest);
        }
    }


    inline void scatter(torch::Tensor &tensor, int root = 0) {
        int rank = this->rank();
        if (rank == root) {
            std::vector<torch::Tensor> chunks = tensor.tensor_split(this->size(), 0);
            for (int i = 0; i < this->size(); ++i) {
                if (i == root) {
                    tensor = chunks[i].clone();
                } else {
                    send(chunks[i], i);
                }
            }
        } else {
            recv(tensor, root);
        }
    }

    inline void gather(torch::Tensor &tensor,
                       std::vector<torch::Tensor> &gathered_tensors, int root = 0) {
        if (this->rank() == root) {
            gathered_tensors.clear();
            gathered_tensors.reserve(this->size());
            gathered_tensors.push_back(tensor);

            for (int src = 0; src < this->size(); ++src) {
                if (src == root) continue;
                torch::Tensor temp = torch::zeros_like(tensor);
                recv(temp, src);
                gathered_tensors.push_back(temp);
            }
        } else {
            send(tensor, root);
        }
    }

    inline boost::mpi::request isend(torch::Tensor &tensor, int dest) {
        SerializedTensor tens{tensor};
        return world_ptr->isend(dest, 0, tens);
    }

    inline boost::mpi::request irecv(torch::Tensor &tensor, int source) {
        SerializedTensor received{torch::zeros_like(tensor)};
        return world_ptr->irecv(source, 0, received);
    }

    inline void barrier() {
        world_ptr->barrier();
    }
};

#endif
