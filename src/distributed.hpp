#pragma once
#ifndef DISTRIBUTED_DL
#define DISTRIBUTED_DL

#include <torch/torch.h>

#include "nccl_distr.cuh"
// #include "serializedtensor.hpp"

// #define SerializedTensor SerializedTensorCPU_impl

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

  inline void send(torch::Tensor &tensor, const int dest) {
    switch (tensor.scalar_type()) {
      case torch::kFloat32: {
        world_ptr->send(dest, 0, tensor.data_ptr<float>(), tensor.numel());
        break;
      }
      case torch::kFloat64: {
        world_ptr->send(dest, 0, tensor.data_ptr<double>(), tensor.numel());
        break;
      }
      case torch::kInt32: {
        world_ptr->send(dest, 0, tensor.data_ptr<int32_t>(), tensor.numel());
        break;
      }
      case torch::kInt64: {
        world_ptr->send(dest, 0, tensor.data_ptr<int64_t>(), tensor.numel());
        break;
      }
      case at::kBool: {
        world_ptr->send(dest, 0, tensor.data_ptr<bool>(), tensor.numel());
        break;
      }
      default:
        throw std::runtime_error("Unsupported ScalarType");
    }
  }

  inline void recv(torch::Tensor &tensor, int source) {
    switch (tensor.scalar_type()) {
      case torch::kFloat32: {
        world_ptr->recv(source, 0, tensor.mutable_data_ptr<float>(), tensor.numel());
        break;
      }
      case torch::kFloat64: {
        world_ptr->recv(source, 0, tensor.mutable_data_ptr<double>(), tensor.numel());
        break;
      }
      case torch::kInt32: {
        world_ptr->recv(source, 0, tensor.mutable_data_ptr<int32_t>(), tensor.numel());
        break;
      }
      case torch::kInt64: {
        world_ptr->recv(source, 0, tensor.mutable_data_ptr<int64_t>(), tensor.numel());
        break;
      }
      case at::kBool: {
        world_ptr->recv(source, 0, tensor.mutable_data_ptr<bool>(), tensor.numel());
        break;
      }
      default:
        throw std::runtime_error("Unsupported ScalarType");
    }
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
#ifdef BOOST_ALLREDUCE_IMPL
    switch (tensor.scalar_type()) {
      case torch::kFloat32: {
        boost::mpi::all_reduce(*world_ptr, boost::mpi::inplace_t<float *>(tensor.mutable_data_ptr<float>()),
                               tensor.numel(), std::plus<float>());
        break;
      }
      case torch::kFloat64: {

        boost::mpi::all_reduce(*world_ptr, boost::mpi::inplace_t<double *>(tensor.mutable_data_ptr<double>()),
                               tensor.numel(), std::plus<double>());
        break;
      }
      case torch::kInt32: {

        boost::mpi::all_reduce(*world_ptr, boost::mpi::inplace_t<int32_t *>(tensor.mutable_data_ptr<int32_t>()),
                               tensor.numel(), std::plus<int32_t>());
        break;
      }
      case torch::kInt64: {

        boost::mpi::all_reduce(*world_ptr, boost::mpi::inplace_t<int64_t *>(tensor.mutable_data_ptr<int64_t>()),
                               tensor.numel(), std::plus<int64_t>());
        break;
      }
      default:
        throw std::runtime_error("Unsupported ScalarType");
    }
    return;
#endif

    int rank = this->rank();
    int size = this->size();
    torch::Tensor result{torch::zeros_like(tensor)};


    if (dest == rank) {
      std::vector received_tensors(size, torch::zeros_like(result));
      received_tensors[dest] = tensor;

      std::vector<boost::mpi::request> requests;
      requests.reserve(size);
      for (int i = 0; i < size; ++i) {
        if (i != dest) {
          requests.emplace_back(irecv(received_tensors[i], i));
        }
      }
      for (auto &request: requests) {
        request.wait();
      }
      torch::Tensor concatenated = torch::stack(received_tensors, /*dim=*/0);

      result = concatenated.sum(0);


      std::vector<boost::mpi::request> send_requests(size - 1);
      for (int i = 0; i < size; ++i) {
        if (i != dest) {
          send_requests.emplace_back(isend(result, i));
        }
      }

      for (auto &request: send_requests) {
        request.wait();
      }
    } else {
      send(tensor, dest);
      recv(result, dest);
    }
    tensor = result;
  }

  inline double default_all_reduce(double precision) {
    double mean_precision;
    boost::mpi::all_reduce(*world_ptr, precision, mean_precision, std::plus<double>());

    return mean_precision / this->size();
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
        if (src == root)
          continue;
        torch::Tensor temp = torch::zeros_like(tensor);
        recv(temp, src);
        gathered_tensors.push_back(temp);
      }
    } else {
      send(tensor, root);
    }
  }

  inline boost::mpi::request isend(torch::Tensor &tensor, int dest) {
    switch (tensor.scalar_type()) {
      case torch::kFloat32: {
        return world_ptr->isend(dest, 0, tensor.data_ptr<float>(), tensor.numel());
      }
      case torch::kFloat64: {
        return world_ptr->isend(dest, 0, tensor.data_ptr<double>(), tensor.numel());
      }
      case torch::kInt32: {
        return world_ptr->isend(dest, 0, tensor.data_ptr<int32_t>(), tensor.numel());
      }
      case torch::kInt64: {
        return world_ptr->isend(dest, 0, tensor.data_ptr<int64_t>(), tensor.numel());
      }
      case at::kBool: {
        return world_ptr->isend(dest, 0, tensor.data_ptr<bool>(), tensor.numel());
      }
      default:
        throw std::runtime_error("Unsupported ScalarType");
    }
  }


  inline boost::mpi::request irecv(torch::Tensor &tensor, int source) {
    switch (tensor.scalar_type()) {
      case torch::kFloat32: {
        return world_ptr->irecv(source, 0, tensor.mutable_data_ptr<float>(), tensor.numel());
      }
      case torch::kFloat64: {
        return world_ptr->irecv(source, 0, tensor.mutable_data_ptr<double>(), tensor.numel());
      }
      case torch::kInt32: {
        return world_ptr->irecv(source, 0, tensor.mutable_data_ptr<int32_t>(), tensor.numel());
      }
      case torch::kInt64: {
        return world_ptr->irecv(source, 0, tensor.mutable_data_ptr<int64_t>(), tensor.numel());
      }
      case at::kBool: {
        return world_ptr->irecv(source, 0, tensor.mutable_data_ptr<bool>(), tensor.numel());
      }
      default:
        throw std::runtime_error("Unsupported ScalarType");
    }
  }


  inline void barrier() {
    world_ptr->barrier();
  }
};

class DistributedTrainer {
  std::unique_ptr<ProcessGroupMPI> pg_mpi = nullptr;
  std::unique_ptr<ProcessGroupNCCL> pg_nccl = nullptr;
  bool cuda = false;

public
:
  [[nodiscard]] int size() const {
    if (cuda) {
      return pg_nccl->size();
    }
    return pg_mpi->size();
  }

  [[nodiscard]]

  int rank() const {
    if (cuda) {
      return pg_nccl->rank();
    }
    return pg_mpi->rank();
  }

  DistributedTrainer(int argc, char **argv, bool is_cuda = 0): cuda{is_cuda} {
    if (is_cuda) {
      pg_nccl = std::make_unique<ProcessGroupNCCL>(argc, argv);
    } else {
      pg_mpi = std::make_unique<ProcessGroupMPI>(argc, argv);
    }
  }

  void synchronize_batch_norms(torch::nn::Module &module) {
    for (auto &child: module.named_children()) {
      if (auto bn = dynamic_cast<torch::nn::BatchNorm2dImpl *>(child.value().get())) {
        if (cuda) {
          pg_nccl->all_reduce(bn->running_mean);
          pg_nccl->all_reduce(bn->running_var);

          bn->running_mean /= pg_nccl->size();
          bn->running_var /= pg_nccl->size();
        } else {
          pg_mpi->all_reduce(bn->running_mean);
          pg_mpi->all_reduce(bn->running_var);

          bn->running_mean /= pg_mpi->size();
          bn->running_var /= pg_mpi->size();
        }
        synchronize_batch_norms(*child.value());
      }
    }
  }

  template<typename T>
  void sync_train_step(torch::nn::ModuleHolder<T> model) {
    if (cuda) {
      for (auto &param: model->named_parameters()) {
        auto meow = param.value().mutable_grad();

        pg_nccl->all_reduce(meow);
        meow.data() = meow.data() / pg_nccl->size();
      }

      for (auto &param: model->named_parameters()) {
        param.value().grad().data() = param.value().grad().data() / pg_nccl->size();
      }
    } else {
      for (auto &param: model->named_parameters()) {
        auto meow = param.value().mutable_grad();

        pg_mpi->all_reduce(meow);
        meow.data() = meow.data() / pg_mpi->size();
      }
    }
  }

  double sync_precision(double precision) {
    return cuda ? pg_nccl->default_all_reduce(precision) : pg_mpi->default_all_reduce(precision);
  }
};

#endif
